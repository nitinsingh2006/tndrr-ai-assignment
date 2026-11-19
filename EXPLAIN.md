# EXPLAIN.md — Interview Cheat Sheet
## tndrr-ai-assignment: Nitin Singh, B.Tech CSE

> Plain English. Short sentences. For live interview prep.

---

## TASK 1: LoRA Fine-tuning

### What the Code Does (Step by Step)

**`generate_dataset.py`**  
1. We pick random combinations of action (order/buy/book), item (laptop/chair), quantity (1-50), and date.  
2. We plug them into ~10 sentence templates like "Please order 5 laptops by Jan 15, 2025."  
3. The correct JSON output is `{"action": "order", "item": "laptop", "quantity": "5", "date": "2025-01-15"}`.  
4. We shuffle everything with a fixed seed (42) so results are reproducible.  
5. First 1000 go to train, last 100 go to test. No overlap — they come from different random seeds.

**`train.py`**  
1. Load a small pre-trained model: SmolLM2-360M-Instruct (360M parameters).  
2. Attach LoRA adapters to the attention layers (q_proj and v_proj). These are small extra weight matrices.  
3. Train only the LoRA weights — the original model weights are frozen (not changed).  
4. After 3 epochs, the model learns to output JSON when given plain English.  
5. Save the adapter weights (tiny, ~5 MB) separately from the base model.

**`evaluate.py`**  
1. Load the base model (no fine-tuning) and the fine-tuned model (base + LoRA adapter).  
2. For each of the 100 test examples, generate an output from both models.  
3. Try to parse each output as JSON.  
4. Compare parsed JSON to ground truth — count exact matches and per-field matches.  
5. Save everything to `results.json`.

### Key Design Choices

| Setting | Value | Why |
|---------|-------|-----|
| LoRA rank (r) | 8 | Enough capacity for 4-field JSON without wasting params |
| Trainable params | 819,200 (0.226%) | Only 0.226% of 362M total model weights are trained (rest frozen) |
| lora_alpha | 16 | = 2×r, standard scaling heuristic from the original LoRA paper |
| target_modules | q_proj, v_proj | Attention projections capture format-following behavior |
| lora_dropout | 0.05 | Slight regularization on a 1000-sample dataset |
| learning_rate | 2e-4 | Standard for LoRA; too high = unstable, too low = slow |
| epochs | 3 | Enough to learn the format; more risks memorizing exact examples |
| batch_size | 2 | Fits RTX 2050 4GB VRAM; grad_accum=8 makes effective batch = 16 |

### Real Results (RTX 2050, 4 GB, ~5 min training)

**Standard test (100 examples):**
| Metric | Base (raw prompt) | Base (zero-shot prompt) | Fine-tuned (LoRA) |
|--------|-------------------|-------------------------|-------------------|
| Valid JSON rate | 0.0% | 100.0% | **100.0%** |
| Exact match | 0.0% | 16.0% | **99.0%** |
| Field: action | 0.0% | 70.0% | **100.0%** |
| Field: item | 0.0% | 27.0% | **99.0%** |
| Field: quantity | 0.0% | 75.0% | **100.0%** |
| Field: date | 0.0% | 71.0% | **100.0%** |

*Why Base scored 0.0% on raw prompt:* The raw test prompt (`<|im_start|>user\nUrgent: get 16 staplers...<|im_end|>`) has no instruction asking for JSON. The base instruct model naturally answers conversationally ("I'm an AI, I cannot purchase staplers..."). When given an explicit zero-shot instruction prompt describing the JSON schema, Base reaches 100% valid JSON but only **16.0% exact match** (failing on plural forms like "16 staplers" and formatting). Fine-tuning taught the model to output exact structured JSON directly without prompt overhead at **99.0% exact match**.

**Hard test (20 adversarial examples — typos, ambiguous phrasing):**
| Metric | Base (raw) | Fine-tuned (LoRA) |
|--------|------------|-------------------|
| Valid JSON rate | 0.0% | **100.0%** |
| Exact match | 0.0% | **70.0%** |
| Field: action | 0.0% | **75.0%** |
| Field: item | 0.0% | **90.0%** |
| Field: quantity | 0.0% | **100.0%** |
| Field: date | 0.0% | **90.0%** |

### Real Failure Analysis

1. **Plural leakage (standard test):** Input: `"purchase 24 speakers"` → model outputs `"item": "speakers"` instead of `"speaker"`. The input uses plural; model copies it verbatim.
2. **Typo propagation (hard test):** Input: `"i need to purhcase 13 mouse"` → model outputs `"action": "purhcase"` (copying the typo) instead of normalizing to `"purchase"`.
3. **Ground-Truth Label Ambiguity (hard test):** Input: `"We need 20 keyboards by 2025-09-08."` → Expected: `"action": "buy"` | Got: `"action": "place"`. The input contains NO explicit action verb (no "buy", "order", or "procure"), only the modal "need". Ground truth arbitrarily assigned `"buy"`, while the model predicted `"place"` (short for place order). This is a genuine dataset ambiguity, not a model reasoning failure.

### GPU Memory Math (7B Model)

**Full fine-tuning:**  
- Weights: 7B × 2 bytes = 14 GB  
- Gradients: another 14 GB (or 28 GB in fp32)  
- Adam optimizer states (m + v): 2 × 7B × 4 bytes = 56 GB  
- Total: ~94 GB → needs 2× A100 80GB  

**LoRA:**  
- Frozen fp16 weights: 14 GB  
- Only train ~40M LoRA params → optimizer states: ~0.5 GB  
- Total: ~16–17 GB → fits on A100 40GB  

**QLoRA:**  
- 4-bit weights: 7B × 0.5 bytes = 3.5 GB  
- LoRA adapters + optimizer: ~0.4 GB  
- Total: ~5 GB → fits on RTX 3090 24GB!  

---

## TASK 2: RAG Pipeline from Scratch

### What the Code Does (Step by Step)

**`download_doc.py`**  
Downloads a 47-page PDF survey about Large Language Models from arXiv (arXiv:2307.06435).

**`rag_pipeline.py`**  
1. **Extract text:** Use `pdfplumber` to pull raw text from each page of the PDF.  
2. **Chunk:** Split text into overlapping windows of ~200 or ~500 tokens. Overlap = 50 tokens. This avoids cutting important sentences at chunk boundaries.  
3. **Embed:** Pass each chunk through `all-MiniLM-L6-v2` (a small sentence-transformer) to get a 384-dim vector.  
4. **Index:** Store all chunk vectors in FAISS (a fast nearest-neighbor search library).  
5. **Retrieve:** For a query, embed it, find the top-5 closest chunks by cosine similarity.  
6. **Answer (optional):** Pass the top-5 chunks as context to an LLM (OpenAI API if key set).

**BM25 (bonus):**  
BM25 is a keyword-matching algorithm. We count word frequencies (with term frequency + inverse document frequency weighting) and score chunks by how many query words they contain. We combine BM25 scores with embedding similarity scores using reciprocal rank fusion (RRF).

**Key bug fixed during audit:** The original BM25 implementation stored *normalized* TF values but tried to recover raw counts during search — introducing floating-point drift and making BM25 give near-identical scores to all chunks (Recall@5 was stuck at 0.333 for both chunk sizes). Fixed by storing raw counts directly.

**Evaluation:**  
For each of 15 test questions: check if the correct page is among the top-5 retrieved chunks → that's Recall@5. MRR = 1/rank of first correct result (averaged over all questions).

### Why BM25 Helps Embedding Search

Embedding search is good at **semantic similarity** — "automobile" matches "car."  
But it fails on **rare words** — if the query has a specific acronym like "RLHF" or a model name like "CLIP" (used in multimodal LLMs), embeddings might miss the exact chunk.  
BM25 is a keyword search — it matches exact terms. Combining both (via Reciprocal Rank Fusion) gives you the best of both worlds.

**Real results (chunk=200 setting):**  
- Dense Recall@1: 0.333 / Recall@5: **0.733** / MRR: 0.517  
- BM25 Hybrid Recall@5: **0.933**  

**Real results (chunk=500 setting):**  
- Dense Recall@1: 0.533 / Recall@5: 0.667 / MRR: 0.600  
- BM25 Hybrid Recall@5: 0.933  

**Best setting: chunk=200** wins on Recall@5 (0.733 vs 0.667) — the primary retrieval metric.  
Why chunk=200 wins: smaller chunks contain exactly one focused passage, so the embedding captures a tight semantic signal. Chunk=500 packs too many topics into one chunk — when a 500-token chunk spans both RLHF and benchmarks, neither query finds it at the top. Chunk=200 keeps each passage in a dedicated chunk, improving Recall@5 by +0.067.  
Hit logic: a chunk counts as a hit if its source page falls in the correct range **OR** passage containment ≥ 50%. Containment = `|passage_words ∩ chunk_words| / |passage_words|` — different from Jaccard, which divides by the union. We use containment because a large chunk surrounding the passage should score 1.0 regardless of extra words.

**Important — honest finding about the BM25 bug:** The old BM25 code (storing normalized TF) actually produced reasonable score spreads (2–4 points across top-5), NOT near-identical scores as initially claimed. The main cause of the original 0.333 Recall@5 was **wrong page ranges in questions.json** (pages 70–100 on a 47-page PDF → zero possible hits). The BM25 code fix is correct (raw counts are more numerically stable), but the page-range fix deserves most of the credit for the metric jump. Fixing both is the right approach.

### What a Re-ranker Does

Embedding retrieval is fast but coarse — it compares query and chunk separately.  
A **re-ranker** (cross-encoder) looks at the query AND chunk together, so it understands their interaction.  
Example: "What year was GPT-4 released?" + a chunk about GPT-4 → cross-encoder scores this pair much better than a chunk that just mentions "GPT" generically.  
Re-ranking is slower but more accurate — typically applied to the top-20 to pick the best 5.

---

## TASK 3: Local LLM Optimisation

### What the Code Does (Step by Step)

**`benchmark.py`**  
1. Load a GGUF model file using `llama-cpp-python`.  
2. Send 10 prompts, one at a time, measuring time before and after each call.  
3. Compute tokens per second = (tokens generated) / (time taken).  
4. Record peak RAM usage using `psutil`.  
5. Save all results to `results.json` and generate a side-by-side comparison in `outputs.md`.

**Two quantizations compared:**  
- **Q4_K_M**: 4-bit with K-means quantization. File size: 491 MB. Peak RAM: 657 MB. Avg speed: **34.6 tok/s**.  
- **Q8_0**: 8-bit weights. Almost no quality loss. File size: 676 MB. Peak RAM: 828 MB. Avg speed: **31.3 tok/s**.  

Both use the same base model (Qwen2.5-0.5B-Instruct), same 10 prompts, CPU-only (no GPU for inference).

### What the KV Cache Is

When a model processes text, at each layer it computes "Keys" and "Values" for the attention mechanism.  
For short prompts this is fine, but for long conversations it means recalculating everything each time.  
The **KV cache** stores the computed Keys and Values for previously seen tokens in RAM.  
This means: for each new token, we only compute attention for THAT token, not the whole history.  
Result: generation is fast after the first pass. Without KV cache, long sequences would be impossibly slow.

### Why Quantisation Saves Memory

A model weight in fp32 uses 4 bytes. In fp16, it's 2 bytes. In 4-bit (Q4), it's 0.5 bytes.  
Quantization maps many original values to a smaller set of discrete values.  
Example: instead of storing "0.3847265," you store "index 7 of 16 possible values."  
You lose some precision, but for most tasks (chat, QA, summarization) the quality drop is very small — often < 2% on benchmarks.

---

## 10 Likely Interview Questions with Short Answers

**Q1: What is LoRA and how does it reduce training cost?**  
LoRA adds small trainable matrices (rank r) to attention layers. Instead of updating all 7B parameters, you only train a few million. This saves memory (no large gradient tensors for frozen weights) and time.

**Q2: What is the difference between LoRA and QLoRA?**  
LoRA keeps the base model in fp16. QLoRA goes further — it loads the base model in 4-bit (NF4 format), so it uses 4× less memory for the base. LoRA adapters are still in fp16. Together, a 7B model that needs 14 GB in LoRA needs only ~5 GB in QLoRA.

**Q3: Why do we use rank r=8 and not r=1 or r=64?**  
Rank controls how much capacity the adapter has. r=1 is too small — it may not learn the format well. r=64 is too large — it wastes memory and may overfit on small datasets. r=8 is a standard sweet spot for simple tasks like structured output.

**Q4: What is catastrophic forgetting?**  
When you fine-tune a model on new data, it may forget what it knew before. LoRA reduces this because the original weights are frozen — only the small adapters change.

**Q5: Why use FAISS for RAG instead of just NumPy?**  
NumPy cosine search is O(n) — it compares every chunk. FAISS uses an inverted index or HNSW graph to search in O(log n) or O(1). For 10,000 chunks it's fine either way, but FAISS scales to millions.

**Q6: What is chunk overlap and why do we use it?**  
Overlap means the end of chunk N is the start of chunk N+1 (e.g., 50 tokens). This prevents a sentence from being split across chunks and disappearing from retrieval. Without overlap, context at chunk boundaries is lost.

**Q7: What is MRR and why is it a better metric than Recall@5?**  
MRR (Mean Reciprocal Rank) = average of 1/rank of the first correct result. If correct is at rank 1, score = 1. If at rank 3, score = 0.33. Recall@5 just says "was it anywhere in top 5?" MRR rewards putting the right answer higher up. In our run: chunk=200 got Recall@5=0.733, MRR=0.517 — chunk=200 is the best setting.

**Q8: What is quantization and what are the tradeoffs?**  
Quantization reduces weight precision (fp32→int8→int4). Saves memory proportionally. The tradeoff: slight quality degradation. Q4 models are ~2× smaller than Q8, but may hallucinate more on complex reasoning. For simple tasks (chat, QA), Q4 is usually acceptable.

**Q9: What is continuous batching (vLLM) and why does it help serving?**  
Traditional serving pads all requests in a batch to the same length — wastes computation. vLLM uses continuous batching: when one request finishes, a new one is inserted without waiting for the whole batch. This maximizes GPU utilization and throughput.

**Q10: Why is the KV cache size per user proportional to context length?**  
KV cache stores K and V tensors for every token in the context, for every layer, for every KV head.  
Formula: `2 × layers × kv_heads × head_dim × context_len × bytes`.  
For Qwen2.5-0.5B (our actual model): `2 × 24 × 2 × 64 × 2048 × 2 = 25 MB per user`.  
20 users → 0.503 GB total KV. For a 7B model (28 layers, 4 KV heads, head_dim=128): 0.118 GB/user, 2.36 GB for 20 users.  
Longer context = linearly more cache. This is why long contexts need paged attention (vLLM) to avoid memory fragmentation.
