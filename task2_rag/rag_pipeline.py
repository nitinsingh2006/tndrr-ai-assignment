"""
RAG Pipeline from Scratch
=========================
No LangChain, No LlamaIndex.
Stack: pdfplumber/PyPDF2 -> chunking -> sentence-transformers -> FAISS/numpy -> (optional) LLM

Features:
  - PDF text extraction with pdfplumber (fallback: PyPDF2)
  - Fixed-size token-level chunking with configurable overlap
  - Dense embedding via sentence-transformers (all-MiniLM-L6-v2)
  - Vector index: FAISS IVF-flat (fallback: numpy cosine similarity)
  - BM25 from scratch (no library)
  - Hybrid retrieval via Reciprocal Rank Fusion (RRF)
  - Evaluation: Recall@1, Recall@5, MRR for two chunk sizes
  - Optional LLM answer generation (OpenAI API)
"""

import os
import json
import re
import math
import time
import sys
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports with graceful fallbacks
# ---------------------------------------------------------------------------

# --- PDF extraction ---
try:
    import pdfplumber

    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2

    _HAS_PYPDF2 = True
except ImportError:
    _HAS_PYPDF2 = False

if not _HAS_PDFPLUMBER and not _HAS_PYPDF2:
    raise ImportError("Install pdfplumber or PyPDF2: pip install pdfplumber PyPDF2")

# --- FAISS ---
try:
    import faiss

    USE_FAISS = True
except ImportError:
    USE_FAISS = False

# --- Sentence Transformers ---
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# 1. PDF Text Extraction
# ---------------------------------------------------------------------------


def extract_text_by_page(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of (page_number, page_text) tuples, 1-indexed page numbers.
    """
    pages: list[tuple[int, str]] = []

    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append((i, text.strip()))
            print(f"[pdfplumber] Extracted {len(pages)} pages from {pdf_path}")
            return pages
        except Exception as exc:
            print(f"[pdfplumber] Failed: {exc}. Trying PyPDF2 …")

    if _HAS_PYPDF2:
        try:
            with open(pdf_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append((i, text.strip()))
            print(f"[PyPDF2] Extracted {len(pages)} pages from {pdf_path}")
            return pages
        except Exception as exc:
            raise RuntimeError(f"Both PDF libraries failed. Last error: {exc}") from exc

    raise RuntimeError("No PDF library available.")


# ---------------------------------------------------------------------------
# 2. Text Chunking
# ---------------------------------------------------------------------------

_APPROX_CHARS_PER_TOKEN = 4  # rough heuristic: 1 token ≈ 4 characters


def _approx_tokens(text: str) -> int:
    """Estimate token count for a string using character heuristic."""
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Split page texts into overlapping chunks measured in approximate tokens.

    Args:
        pages: List of (page_number, text) tuples.
        chunk_size: Target chunk size in approximate tokens (1 token ≈ 4 chars).
        overlap: Overlap between consecutive chunks in approximate tokens.

    Returns:
        List of chunk dicts with keys: chunk_id, page, text, token_count.
    """
    chunk_chars = chunk_size * _APPROX_CHARS_PER_TOKEN
    overlap_chars = overlap * _APPROX_CHARS_PER_TOKEN
    step_chars = max(1, chunk_chars - overlap_chars)

    chunks: list[dict] = []
    chunk_id = 0

    for page_num, text in pages:
        if not text.strip():
            continue

        # Slide a window over the page text
        start = 0
        while start < len(text):
            end = start + chunk_chars
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": chunk_str,
                        "token_count": _approx_tokens(chunk_str),
                    }
                )
                chunk_id += 1
            start += step_chars

    print(
        f"[chunking] chunk_size={chunk_size} tokens → {len(chunks)} chunks "
        f"(overlap={overlap} tokens)"
    )
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding Index
# ---------------------------------------------------------------------------


def build_index(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[np.ndarray, SentenceTransformer, Any]:
    """Embed chunks and build a FAISS index (or return embeddings for numpy fallback).

    Args:
        chunks: List of chunk dicts (must contain 'text' key).
        model_name: HuggingFace sentence-transformers model name.

    Returns:
        Tuple of (embeddings array [N, D], loaded model, faiss_index or None).
    """
    print(f"[index] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"[index] Encoding {len(texts)} chunks …")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # unit vectors → cosine ≡ dot product
        convert_to_numpy=True,
    )
    print(f"[index] Encoded in {time.time() - t0:.1f}s. Shape: {embeddings.shape}")

    index = None
    if USE_FAISS:
        dim = embeddings.shape[1]
        # Inner-product index on L2-normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        print(f"[index] FAISS IndexFlatIP built with {index.ntotal} vectors (dim={dim})")
    else:
        print("[index] FAISS not available — will use numpy cosine similarity")

    return embeddings, model, index


# ---------------------------------------------------------------------------
# 4. Dense Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    index: Any,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the top_k most relevant chunks for a query.

    Args:
        query: User query string.
        chunks: Original chunk list (for metadata).
        embeddings: Pre-computed chunk embeddings [N, D].
        model: Loaded SentenceTransformer model.
        index: FAISS index or None (uses numpy fallback).
        top_k: Number of results to return.

    Returns:
        List of dicts: chunk metadata + 'score' (cosine similarity) + 'rank'.
    """
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)  # shape [1, D]

    if USE_FAISS and index is not None:
        scores, indices = index.search(q_emb, min(top_k, len(chunks)))
        scores = scores[0].tolist()
        indices = indices[0].tolist()
    else:
        # Numpy cosine similarity (embeddings already L2-normalised)
        sims = (embeddings @ q_emb.T).squeeze()  # [N]
        top_indices = np.argsort(sims)[::-1][:top_k].tolist()
        indices = top_indices
        scores = sims[top_indices].tolist()

    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        result = dict(chunks[idx])
        result["score"] = float(score)
        result["rank"] = rank
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 5. BM25 from Scratch
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_bm25_index(chunks: list[dict]) -> dict:
    """Pre-compute IDF and term frequencies for BM25.

    Returns a dict with keys: 'tf', 'idf', 'avgdl', 'N'.
    """
    N = len(chunks)
    tf: list[dict[str, float]] = []
    df: dict[str, int] = {}

    dl_list: list[int] = []
    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        dl_list.append(len(tokens))
        freq: dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
        # Store RAW counts (not normalised) — BM25 formula handles length norm
        tf.append(freq)
        for tok in freq:
            df[tok] = df.get(tok, 0) + 1

    avgdl = sum(dl_list) / max(1, N)
    idf: dict[str, float] = {}
    for tok, n in df.items():
        idf[tok] = math.log((N - n + 0.5) / (n + 0.5) + 1)

    return {"tf": tf, "idf": idf, "avgdl": avgdl, "N": N, "dl": dl_list}


def bm25_search(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    _bm25_cache: dict | None = None,
) -> list[dict]:
    """BM25 keyword retrieval from scratch.

    Args:
        query: User query string.
        chunks: Corpus chunks with 'text' key.
        top_k: Number of results to return.
        k1: Term saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).
        _bm25_cache: Optional pre-built BM25 index dict (avoids recomputation).

    Returns:
        List of chunk dicts with 'score' (BM25) and 'rank'.
    """
    if _bm25_cache is None:
        _bm25_cache = _build_bm25_index(chunks)

    idf = _bm25_cache["idf"]
    tf_list = _bm25_cache["tf"]   # list of {tok: raw_count}
    avgdl = _bm25_cache["avgdl"]
    dl_list = _bm25_cache["dl"]  # list of doc lengths (raw token counts)

    q_tokens = _tokenize(query)

    scores = np.zeros(len(chunks))
    for tok in q_tokens:
        if tok not in idf:
            continue
        idf_val = idf[tok]
        for i, tf_map in enumerate(tf_list):
            tf_val = float(tf_map.get(tok, 0))   # raw count
            dl = dl_list[i]
            numerator = tf_val * (k1 + 1)
            denominator = tf_val + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf_val * (numerator / max(1e-9, denominator))

    top_indices = np.argsort(scores)[::-1][:top_k].tolist()
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        result = dict(chunks[idx])
        result["score"] = float(scores[idx])
        result["rank"] = rank
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 6. Hybrid Retrieval (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------


def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    index: Any,
    top_k: int = 5,
    rrf_k: int = 60,
    bm25_cache: dict | None = None,
) -> list[dict]:
    """Fuse dense and BM25 rankings using Reciprocal Rank Fusion.

    RRF score(d) = Σ_r  1 / (rrf_k + rank_r(d))

    Args:
        query: User query string.
        chunks: Corpus chunks.
        embeddings: Pre-computed dense embeddings.
        model: SentenceTransformer model.
        index: FAISS index or None.
        top_k: Final number of results to return.
        rrf_k: RRF constant (default 60, from the original RRF paper).
        bm25_cache: Pre-built BM25 index (avoids recomputation).

    Returns:
        Top_k chunks sorted by RRF score, with 'score' and 'rank' keys.
    """
    # Retrieve from both systems — use a larger pool for fusion
    pool = min(len(chunks), top_k * 4)

    dense_results = retrieve(query, chunks, embeddings, model, index, top_k=pool)
    bm25_results = bm25_search(
        query, chunks, top_k=pool, _bm25_cache=bm25_cache
    )

    rrf_scores: dict[int, float] = {}

    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # Build a lookup from chunk_id -> chunk dict
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    results = []
    for rank, cid in enumerate(sorted_ids, start=1):
        if cid not in chunk_lookup:
            continue
        result = dict(chunk_lookup[cid])
        result["score"] = rrf_scores[cid]
        result["rank"] = rank
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 7. Optional LLM Generation
# ---------------------------------------------------------------------------


def answer_with_llm(
    query: str,
    context_chunks: list[dict],
    api_key: str | None = None,
) -> str:
    """Generate an answer using the OpenAI API (skipped if no API key).

    Args:
        query: User query.
        context_chunks: Retrieved chunk dicts (use 'text' key for context).
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

    Returns:
        Generated answer string, or 'LLM_SKIPPED' if no API key available.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "LLM_SKIPPED"

    try:
        import urllib.request as _req
        import json as _json

        context = "\n\n".join(
            f"[Page {c['page']}]\n{c['text']}" for c in context_chunks[:5]
        )
        prompt = (
            f"Based on the following context, answer the question concisely.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        payload = _json.dumps(
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.0,
            }
        ).encode()
        request = _req.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with _req.urlopen(request, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"LLM_ERROR: {exc}"


# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------


def _passage_overlap(passage: str, chunk_text: str) -> float:
    """Return word-level overlap ratio between passage and chunk (Jaccard on words)."""
    if not passage or not chunk_text:
        return 0.0
    p_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', passage.lower()))
    c_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', chunk_text.lower()))
    if not p_words:
        return 0.0
    # What fraction of passage words appear in the chunk?
    return len(p_words & c_words) / len(p_words)


def _is_hit(q: dict, result: dict) -> bool:
    """A chunk is a hit if passage overlap >= 50% OR page in correct_page_range."""
    page_range = q.get("correct_page_range", [0, 999])
    lo, hi = page_range[0], page_range[1]
    page_ok = lo <= result["page"] <= hi
    passage = q.get("correct_source_passage", "")
    overlap = _passage_overlap(passage, result.get("text", ""))
    return page_ok or overlap >= 0.50


def _page_in_results(page_range: list[int], results: list[dict],
                     q: dict | None = None) -> bool:
    """Check if any result is a hit (page match OR passage overlap >= 50%)."""
    if q is not None:
        return any(_is_hit(q, r) for r in results)
    lo, hi = page_range[0], page_range[1]
    return any(lo <= r["page"] <= hi for r in results)


def _reciprocal_rank(page_range: list[int], results: list[dict],
                     q: dict | None = None) -> float:
    """Return 1/rank of the first hit result."""
    for r in results:
        hit = _is_hit(q, r) if q is not None else (page_range[0] <= r["page"] <= page_range[1])
        if hit:
            return 1.0 / r["rank"]
    return 0.0


def evaluate_retrieval(
    questions_path: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    index: Any,
    bm25_cache: dict | None = None,
    top_k: int = 5,
) -> dict:
    """Evaluate Recall@1, Recall@5, and MRR for dense + BM25 hybrid retrieval.

    Runs evaluation for BOTH chunk_size=200 and chunk_size=500 by rebuilding
    the index each time (the caller must pass the appropriate chunk list).

    Args:
        questions_path: Path to questions.json.
        chunks: Current chunk list.
        embeddings: Corresponding embeddings.
        model: SentenceTransformer model.
        index: FAISS index or None.
        bm25_cache: Pre-built BM25 index (optional, built here if None).
        top_k: Retrieval depth (default 5).

    Returns:
        Dict with recall_at_1, recall_at_5, mrr, hybrid_recall_at_5, failures.
    """
    with open(questions_path, "r") as fh:
        questions = json.load(fh)

    if bm25_cache is None:
        print("[eval] Building BM25 index …")
        bm25_cache = _build_bm25_index(chunks)

    recalls_at_1: list[float] = []
    recalls_at_5: list[float] = []
    mrrs: list[float] = []
    hybrid_recalls_at_5: list[float] = []
    failures: list[dict] = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        page_range = q["correct_page_range"]

        # --- Dense retrieval ---
        top5 = retrieve(question, chunks, embeddings, model, index, top_k=5)
        top1 = top5[:1]

        hit_at_1 = float(_page_in_results(page_range, top1, q=q))
        hit_at_5 = float(_page_in_results(page_range, top5, q=q))
        rr = _reciprocal_rank(page_range, top5, q=q)

        recalls_at_1.append(hit_at_1)
        recalls_at_5.append(hit_at_5)
        mrrs.append(rr)

        if hit_at_5 == 0.0:
            failures.append(
                {
                    "id": qid,
                    "question": question,
                    "expected_page_range": page_range,
                    "retrieved_pages": [r["page"] for r in top5],
                    "type": q.get("type", "unknown"),
                }
            )

        # --- Hybrid retrieval ---
        hybrid5 = hybrid_retrieve(
            question,
            chunks,
            embeddings,
            model,
            index,
            top_k=5,
            bm25_cache=bm25_cache,
        )
        hybrid_recalls_at_5.append(float(_page_in_results(page_range, hybrid5, q=q)))

    metrics = {
        "num_chunks": len(chunks),
        "recall_at_1": round(sum(recalls_at_1) / len(recalls_at_1), 4),
        "recall_at_5": round(sum(recalls_at_5) / len(recalls_at_5), 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "hybrid_recall_at_5": round(
            sum(hybrid_recalls_at_5) / len(hybrid_recalls_at_5), 4
        ),
        "failures": failures,
    }

    print(
        f"[eval] recall@1={metrics['recall_at_1']:.3f}  "
        f"recall@5={metrics['recall_at_5']:.3f}  "
        f"MRR={metrics['mrr']:.3f}  "
        f"hybrid_recall@5={metrics['hybrid_recall_at_5']:.3f}  "
        f"failures={len(failures)}/{len(questions)}"
    )
    return metrics


# ---------------------------------------------------------------------------
# 9. Main — Run Full Evaluation and Save results.json
# ---------------------------------------------------------------------------


def _pdf_path() -> str:
    """Return PDF path, searching common locations."""
    candidates = [
        "data/survey_llm.pdf",
        os.path.join(os.path.dirname(__file__), "data", "survey_llm.pdf"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "PDF not found. Run: python download_doc.py  (or place PDF at data/survey_llm.pdf)"
    )


def main() -> None:
    """Main entry point: evaluate two chunk sizes + BM25 hybrid, save results.json."""
    pdf_path = _pdf_path()
    questions_path = os.path.join(os.path.dirname(__file__) or ".", "questions.json")

    print("=" * 60)
    print("RAG Pipeline — Full Evaluation")
    print("=" * 60)

    # -- Extract text --
    print("\n--- Step 1: PDF Extraction ---")
    pages = extract_text_by_page(pdf_path)
    total_pages = len(pages)
    print(f"Total pages: {total_pages}")

    model_name = "all-MiniLM-L6-v2"
    results: dict = {
        "document": {
            "source": "https://arxiv.org/pdf/2307.06435",
            "title": "A Survey of Large Language Models",
            "pages": total_pages,
        },
        "chunk_200": {},
        "chunk_500": {},
        "bm25_hybrid": {},
        "failures": [],
        "best_chunk_size": None,
    }

    all_failures: list[dict] = []

    for chunk_size, label in [(200, "chunk_200"), (500, "chunk_500")]:
        print(f"\n--- Step 2+3: Chunk size = {chunk_size} ---")
        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=chunk_size // 4)

        print(f"\n--- Step 4: Building index (chunk_size={chunk_size}) ---")
        embeddings, model, index = build_index(chunks, model_name=model_name)

        print(f"\n--- Step 5: Evaluating (chunk_size={chunk_size}) ---")
        bm25_cache = _build_bm25_index(chunks)
        metrics = evaluate_retrieval(
            questions_path, chunks, embeddings, model, index, bm25_cache=bm25_cache
        )

        results[label] = {
            "num_chunks": metrics["num_chunks"],
            "recall_at_1": metrics["recall_at_1"],
            "recall_at_5": metrics["recall_at_5"],
            "mrr": metrics["mrr"],
        }
        results["bm25_hybrid"][label + "_hybrid_recall_at_5"] = metrics[
            "hybrid_recall_at_5"
        ]
        all_failures.extend(
            [{"chunk_size": chunk_size, **f} for f in metrics["failures"]]
        )

    # Determine best chunk size by recall@5
    r5_200 = results["chunk_200"].get("recall_at_5", 0)
    r5_500 = results["chunk_500"].get("recall_at_5", 0)
    results["best_chunk_size"] = 200 if r5_200 >= r5_500 else 500
    results["failures"] = all_failures

    # -- Save results --
    out_path = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[results] Saved to {out_path}")

    # -- Print summary table --
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'chunk=200':>12} {'chunk=500':>12}")
    print("-" * 50)
    for metric in ["num_chunks", "recall_at_1", "recall_at_5", "mrr"]:
        v200 = results["chunk_200"].get(metric, "-")
        v500 = results["chunk_500"].get(metric, "-")
        print(f"{metric:<25} {str(v200):>12} {str(v500):>12}")
    print("-" * 50)
    print(
        f"{'hybrid_recall@5 (200)':25} "
        f"{str(results['bm25_hybrid'].get('chunk_200_hybrid_recall_at_5', '-')):>12}"
    )
    print(
        f"{'hybrid_recall@5 (500)':25} "
        f"{str(results['bm25_hybrid'].get('chunk_500_hybrid_recall_at_5', '-')):>12}"
    )
    print(f"\nBest chunk size: {results['best_chunk_size']}")
    print(f"Total retrieval failures: {len(all_failures)}")


if __name__ == "__main__":
    main()
