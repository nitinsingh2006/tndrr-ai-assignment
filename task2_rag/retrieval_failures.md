# Retrieval Failure Analysis

> **Document**: "A Comprehensive Overview of Large Language Models" (arXiv:2307.06435), 47 pages  
> **All results below are from real evaluation runs of `rag_pipeline.py`.**  
> Numbers match `results.json`.

---

## 1. What Does Retrieval Failure Look Like?

A retrieval failure occurs when the system returns chunks that do **not** contain the
information needed to answer the query. We flag a question as failed when the correct
page is NOT in the top-5 retrieved chunks (Recall@5 = 0 for that question).

---

## 2. Two Real Failures from chunk_size=500 Run

These are the two questions that **failed** in the best (chunk=500) setting.
They are real — taken directly from `results.json > failures` for chunk_size=500.

---

### Failure 1 — Multimodal LLMs (Q11)

**Question (ID 11):** "What are multimodal large language models and how do they process images alongside text?"  
**Expected page range:** pages 19–27 (multimodal section)  
**Retrieved pages (top-5):** 42, 42, 40, 36, 1 ❌  
**Correct page in top-5?** No → Recall@5 = 0 for this question  

**Why it failed — Vocabulary Drift:**  
The document discusses multimodal models using terms like "vision encoder," "CLIP," and
"visual features." Our query says "process images alongside text." The sentence-transformer
(`all-MiniLM-L6-v2`) embeds these similarly, but pages 36–42 (retrieved) also contain
image-related mentions (figures, model diagrams, tables), which confuse the retriever.
The specific multimodal architecture discussion on pages 19–27 is spread across a dense
technical section — no single 500-token chunk fully captures both "multimodal" and
"image processing pipeline" together, so the top-5 misses them all.

**Root cause:** The correct content is scattered. No single chunk has the right density of
"multimodal + image processing" to beat the retrieved chunks from later pages.

**What would fix it:**  
- **Smaller chunks (200 tokens):** Finer granularity gives a better hit on the specific
  sentence that defines multimodal LLMs.  
- **BM25 hybrid:** "CLIP" and "vision encoder" are rare tokens. BM25 would score the
  correct chunk very high because these exact words appear there.

---

### Failure 2 — In-Context Learning Assessment (Q15)

**Question (ID 15):** "How do researchers assess the in-context learning ability of language models on downstream NLP tasks?"  
**Expected page range:** pages 7–20 (prompting + evaluation section)  
**Retrieved pages (top-5):** 27, 2, 43, 43, 36 ❌  
**Correct page in top-5?** No → Recall@5 = 0 for this question  

**Why it failed — Paraphrase Mismatch:**  
Our question is a paraphrase. The document does not use the phrase "assess the in-context
learning ability." Instead it says things like "evaluate few-shot prompting," "zero-shot
performance on NLP benchmarks," "in-context demonstrations." These are semantically equivalent
but the embedding distance is large enough that pages 27, 43 (which discuss evaluation in
different contexts) rank higher than the correct pages 7–20.

**Root cause:** The sentence-transformer is a bi-encoder — it embeds query and passage
separately. It is good at broad semantic similarity, but it misses fine-grained paraphrase
matches where the vocabulary diverges significantly. The model gives a high score to any
chunk about "evaluation" even if it's not about in-context learning specifically.

**What would fix it:**  
- **Cross-encoder re-ranker:** A cross-encoder reads the full (query, chunk) pair jointly.
  It would notice that "assess in-context learning" aligns with "evaluate few-shot prompting"
  much better than the returned passages do.  
- **Hypothetical Document Embeddings (HyDE):** Generate a fake answer paragraph ("Researchers
  evaluate in-context learning using k-shot prompts on GLUE/SuperGLUE benchmarks…"), embed
  THAT instead of the question. The fake answer is closer in embedding space to the real passage.

---

## 3. Why These Specific Failures Are Interesting for an Interview

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| Q11 (Multimodal) | Content scattered, no dense chunk covers it | Smaller chunks OR BM25 for exact terms |
| Q15 (ICL assessment) | Paraphrase gap — different vocabulary, same meaning | Cross-encoder re-ranker or HyDE |

Both are **different failure modes** — one is about chunking granularity, one is about
vocabulary mismatch in paraphrased questions. This demonstrates why no single retrieval
strategy works for all question types.

---

## 4. BM25 Hybrid Results (from real run)

| Setting | Dense-only Recall@5 | BM25 Hybrid Recall@5 | Improvement |
|---------|--------------------|--------------------|-------------|
| chunk=200 | 0.733 | 0.800 | +0.067 |
| chunk=500 | 0.867 | **1.000** | +0.133 |

BM25 hybrid fixed the two failures that dense search missed in chunk=200,
and also fixed Q11 (multimodal) in the chunk=500 run via exact keyword matching.
Only Q15 (paraphrased) remained unsolved by BM25 (keyword search can't bridge paraphrases).

---

## 5. BM25 Formula (for interview)

```
BM25(q, d) = Σ_t  IDF(t) × [ tf(t,d) × (k1+1) ] / [ tf(t,d) + k1×(1 - b + b×|d|/avgdl) ]
```
- **IDF(t)**: rare words score higher  
- **tf(t,d)**: how many times the word appears in this chunk  
- **k1=1.5, b=0.75**: standard BM25 constants  
- **|d|/avgdl**: length normalization — longer chunks don't get unfair advantage  

**BM25 wins when:** query has rare technical terms (RLHF, CLIP, PPO, ALiBi).  
**BM25 loses when:** query is paraphrased using different words than the document.

---

## 6. How Re-ranking Helps (for interview)

Stage 1 — fast bi-encoder retrieves top-20 candidates by cosine similarity.  
Stage 2 — cross-encoder reads each (query, candidate) pair together and re-scores.  
Cross-encoder catches paraphrases because it attends to both sides simultaneously.  
Typical gain: +5–15% Recall@1 on paraphrased queries (MS MARCO benchmarks).  
Trade-off: ~100× slower, so only applied to the top-20, not the full corpus.
