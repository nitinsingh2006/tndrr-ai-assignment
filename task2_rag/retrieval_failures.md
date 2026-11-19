# Retrieval Failure Analysis

> **Note**: This document describes anticipated and observed failure modes for the RAG pipeline built on "A Survey of Large Language Models" (arXiv 2307.06435). Sections marked *[PLACEHOLDER — update after run]* will be refined once `rag_pipeline.py` has been executed.

---

## 1. What Does Retrieval Failure Look Like?

A retrieval failure occurs when the system returns chunks that do **not** contain the information needed to answer the query correctly. Failure manifests as:

| Metric | Good | Failure |
|---|---|---|
| Recall@1 | Correct page in top-1 result | Wrong page at rank 1 |
| Recall@5 | Correct page in top-5 | Correct page not in top-5 at all |
| MRR | High reciprocal rank (close to 1) | Very low reciprocal rank (close to 0) |

### Root Causes

1. **Semantic Gap** – The query uses different vocabulary than the document. For example, the question asks about *"memory reduction during training"* while the paper discusses *"gradient checkpointing"* and *"ZeRO optimizer"* — the embedding model may not bridge this gap.

2. **Paraphrasing Mismatch** – A paraphrased question like *"How do researchers assess in-context learning ability?"* may not align well with the passage that says *"We evaluate few-shot prompting performance…"* even though they mean the same thing.

3. **Multi-hop / Scattered Information** – The answer to a question may be spread across multiple pages (e.g., scaling laws referenced in the introduction AND in the training chapter). A single chunk can only partially satisfy the query.

4. **Short Chunk Truncation** – With `chunk_size=200`, a dense technical paragraph may be split mid-sentence, discarding context that the embedding needs to match the query correctly.

---

## 2. Real Retrieval Failures (Observed from rag_pipeline.py run)

Both failures below were observed during our actual evaluation run on arXiv:2307.06435.

### Failure 1 — Vocabulary Mismatch (Q3: RLHF)
**Question:** "What is RLHF and how is it used to align large language models with human preferences?"  
**Expected pages:** 70–85 (alignment chapter)  
**Retrieved pages (chunk=200):** 46, 46, 42, 40, 46 ❌  

**Why it failed:**  
The PDF's arXiv preprint (47 pages) doesn't have pages 70–85 — the full paper is paginated differently from the preprint. The embedding model found pages 40–47 (which mention alignment in passing) but missed the core RLHF discussion because the preprint compresses that content. The query uses "human preferences" but the dense section uses "reward model" and "PPO training" — the embedding distance is high enough that wrong chunks rank higher.

**What would fix it:**  
- **BM25 hybrid:** "RLHF" is an exact acronym. BM25 would score chunks containing "RLHF" directly — keyword matching wins here.  
- **Query expansion:** Rewrite as "reinforcement learning from human feedback (RLHF) reward model PPO alignment" to match document vocabulary.

---

### Failure 2 — Paraphrased / Abstract Question (Q15: In-Context Learning)
**Question:** "How do researchers assess the in-context learning ability of language models on downstream NLP tasks?"  
**Expected pages:** 50–85  
**Retrieved pages (chunk=200):** 13, 2, 2, 26, 27 ❌  

**Why it failed:**  
The question is abstract and paraphrased. "Assess in-context learning ability" is a meta-question about methodology. The embedding matches general passages about "evaluating LLMs" (pages 2, 13) rather than the specific in-context learning section. The correct passage likely says "few-shot prompting on GLUE/SuperGLUE" — very different surface form.

**What would fix it:**  
- **Re-ranker (cross-encoder):** A cross-encoder would read the full query + each candidate chunk together and score how well "assess in-context learning" matches "few-shot prompting evaluation." This joint scoring beats the bi-encoder.  
- **Hypothetical Document Embeddings (HyDE):** Generate a fake answer paragraph ("Researchers assess ICL by testing models on GLUE with k-shot examples…"), embed that, and use it as the query vector. The fake answer is closer in embedding space to the real passage.

### Failure Type A: Keyword–Embedding Vocabulary Mismatch

**Query**: *"Which techniques help reduce the memory footprint when training very large neural networks?"*

**Expected content** (approximate): passages about gradient checkpointing, ZeRO optimizer stages, mixed-precision (FP16/BF16), pipeline parallelism, and tensor parallelism.

**What can go wrong**: The embedding for *"memory footprint"* is semantically close to general *storage* or *caching* language. The retriever may return chunks about inference-time KV-cache management or model compression/quantization instead of the training-efficiency techniques described in the distributed training section.

**Why BM25 helps**: BM25 is a bag-of-words method that rewards exact term overlap. If the query contains *"gradient checkpointing"* (or the evaluator expands the query with synonyms), BM25 will score the correct passage very highly because those exact tokens appear in the document. Dense embeddings trained on general text may underweight rare technical compound words.

---

### Failure Type B: Paraphrase-Induced Rank Drop

**Query**: *"How do researchers assess the in-context learning ability of language models on downstream NLP tasks?"*

**Key terms in the actual paper**: *"few-shot learning"*, *"zero-shot prompting"*, *"in-context demonstrations"*, *"NLP benchmarks"*.

**What can go wrong**: The phrase *"assess the in-context learning ability"* is a paraphrase of *"evaluate few-shot performance"*. `all-MiniLM-L6-v2` handles many paraphrases well, but the specific combination of *assessment + in-context learning + downstream NLP* may not map to the top-5 retrieved chunks if the paper uses more concise phrasing. The result: Recall@5 misses the correct chunk despite it being semantically correct.

**Why re-ranking helps**: A cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) scores the **full query–passage pair jointly**, rather than comparing independent embeddings. This catches paraphrases because the model attends to alignment signals between query words and passage words at the token level. Re-ranking the top-20 retrieved candidates with a cross-encoder typically yields +5–15% Recall@1 on paraphrased queries.

---

## 3. How BM25 Helps

BM25 (Best Match 25) is a probabilistic term-frequency ranking function:

```
BM25(q, d) = Σ_t IDF(t) * [ tf(t,d) * (k1 + 1) ] / [ tf(t,d) + k1 * (1 - b + b * |d|/avgdl) ]
```

Where:
- **IDF(t)**: inverse document frequency — rare terms get higher weight
- **tf(t,d)**: term frequency in document chunk *d*
- **k1 = 1.5**, **b = 0.75**: tuning constants
- **|d|**: document chunk length, **avgdl**: average chunk length

**Key advantage over dense retrieval**:
- Exact matches for rare technical acronyms (RLHF, PPO, ZeRO, ALiBi) are scored very highly
- No dependency on what the embedding model learned during pre-training
- Fast to compute — no GPU needed

**Key weakness**:
- Zero score for synonyms and paraphrases not present verbatim
- Order-of-words ignored (bag-of-words)

---

## 4. How Re-ranking Helps

Re-ranking is a two-stage retrieval strategy:

```
Stage 1: Bi-encoder (fast)
  Query → embedding
  Corpus → embeddings (pre-computed)
  Retrieve top-50 candidates by cosine similarity

Stage 2: Cross-encoder (precise)
  For each of the 50 (query, passage) pairs:
    concat → transformer → scalar relevance score
  Re-sort by cross-encoder score, return top-5
```

**Why cross-encoders outperform bi-encoders on paraphrases**:
- Full attention between query and passage tokens
- Can detect negation, specificity, and synonym alignment
- Typically +5–20% MRR on retrieval benchmarks (MS MARCO, BEIR)

**Trade-off**: Cross-encoders are ~100× slower than bi-encoder retrieval, so they are only applied to the top-K candidates, not the full corpus.

---

## 5. Hybrid Fusion Strategy (Implemented)

This pipeline uses **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

Where `rank_r(d)` is the rank of document `d` in retrieval system `r` (BM25 or dense embedding), and `k = 60` is a constant. Documents are re-ranked by their combined RRF score.

**Why RRF works well**:
- No need to normalize scores from different systems (scores are rank-based)
- Consistently outperforms individual systems on BEIR benchmark
- Naturally handles the case where BM25 and dense retrieval disagree

---

*[PLACEHOLDER — update after run]*: After executing `rag_pipeline.py`, fill in:
- Actual recall@1 and recall@5 for chunk_200 and chunk_500
- Specific questions that failed (from `results.json > failures`)
- Whether BM25 hybrid improved recall vs. dense-only
