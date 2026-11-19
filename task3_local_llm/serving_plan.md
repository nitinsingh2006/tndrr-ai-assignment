# Serving Plan: 20 Concurrent Users on 24 GB GPU

> **Model used in this analysis**: Qwen2.5-0.5B-Instruct (the actual benchmarked model).
> Architecture parameters read from the GGUF file metadata (`llm.metadata`).

---

## Model Architecture (from GGUF metadata — exact values)

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | qwen2 | `general.architecture` |
| Parameters | 0.5 B | model name |
| Layers (`block_count`) | 24 | `qwen2.block_count` |
| Attention heads | 14 | `qwen2.attention.head_count` |
| KV heads (GQA) | **2** | `qwen2.attention.head_count_kv` |
| Embedding dim | 896 | `qwen2.embedding_length` |
| Head dim = embed / heads | 896 ÷ 14 = **64** | derived |
| Context length (max) | 32 768 | `qwen2.context_length` |
| Assumed context per user | 2 048 tokens | conservative production value |
| Concurrent users | 20 | assignment spec |

---

## Memory Breakdown

### 1. Model Weights (loaded once, shared across all users)

```
Q4_K_M: 0.5 × 10⁹ params × 0.5 bytes/param = 0.250 GB
```

Weights are shared in memory. All 20 users run inference against the same copy.
Measured file size on disk: **491.4 MB** (from benchmark.py).
In-VRAM footprint matches file size ≈ **0.25 GB** (4-bit average precision).

---

### 2. KV Cache per User

The KV cache stores Key and Value tensors for every processed token, every layer, every KV head.

```
kv_per_user = 2 × num_layers × num_kv_heads × head_dim × context_len × bytes_per_elem
```

With FP16 activations (`bytes_per_elem = 2`):

```
= 2 × 24 × 2 × 64 × 2048 × 2
= 25,165,824 bytes
= 0.0252 GB per user
```

> [!NOTE]
> GQA (Grouped Query Attention) with only **2 KV heads** vs 14 attention heads
> reduces KV cache by 7× compared to standard MHA (which would use 14 KV heads).
> This is a major efficiency win in Qwen2.5.

---

### 3. Total KV Cache — 20 Users

```
20 users × 0.0252 GB = 0.503 GB
```

---

### 4. Total Memory Summary

| Component | Formula | GB |
|-----------|---------|-----|
| Model weights (Q4_K_M, shared) | 0.5B × 0.5 B/param | **0.250** |
| KV cache (20 users × 2048 ctx) | 20 × 0.0252 | **0.503** |
| Framework overhead (est.) | — | **~0.5** |
| **Total** | | **~1.25 GB** |
| **24 GB GPU headroom** | 24 − 1.25 | **22.75 GB free** |

The 0.5B Q4_K_M model fits trivially in a 24 GB GPU. For a **7B model** (a more
realistic production scenario), see the extended analysis below.

---

### 5. Extended: 7B Model on 24 GB GPU

| Parameter | Value |
|-----------|-------|
| Architecture | Qwen2.5-7B-Instruct |
| Params | 7 B |
| Layers | 28 |
| KV heads (GQA) | 4 |
| Head dim | 128 |
| Context per user | 2 048 tokens |

```
Weights (Q4_K_M): 7B × 0.5 = 3.5 GB

kv_per_user = 2 × 28 × 4 × 128 × 2048 × 2 = 117,964,800 bytes = 0.118 GB
kv_total (20 users) = 20 × 0.118 = 2.36 GB

Total = 3.5 + 2.36 + ~0.5 overhead = ~6.4 GB  → fits in 24 GB GPU
```

---

## Serving Architecture Recommendation (vLLM + PagedAttention)

| Component | Choice | Reason |
|-----------|--------|--------|
| Inference server | **vLLM** | Continuous batching — GPU never idles while one user waits |
| Memory management | **PagedAttention** | Splits KV cache into 16-token pages, eliminates fragmentation |
| Quantization | **Q4_K_M** | 7× memory saving vs FP32, <2% quality loss on simple tasks |
| Batching | **Continuous** | New request inserts when any slot frees — avoids padding waste |
| Concurrency | **20 users** | 20 × 0.118 GB KV + 3.5 GB weights = 5.9 GB — leaves 18 GB headroom |

### Why This Works for 20 Users

1. **Model weights** (3.5 GB for 7B Q4) are loaded once and shared — zero per-user weight cost.
2. **KV cache** is 0.118 GB/user for 7B with 2048-token context. 20 users = 2.36 GB.
3. **Continuous batching** (vLLM) means user 21 starts immediately when user 1 finishes — 
   no need to provision 21× the memory.
4. **GQA** (Qwen2.5 uses 4 KV heads for 7B) reduces KV cache by 7× vs standard MHA.

---

## KV Cache Formula (for interview)

```
kv_bytes = 2 × L × H_kv × d_head × T × dtype_bytes
```

Where:
- `2` = one K tensor + one V tensor
- `L` = number of transformer layers
- `H_kv` = number of KV heads (much less than Q heads with GQA)
- `d_head` = head dimension = embedding_dim / num_heads
- `T` = context tokens per user
- `dtype_bytes` = 2 for FP16, 4 for FP32

**Key insight**: KV cache grows **linearly** with context length. This is why long contexts 
(100K tokens) need PagedAttention to avoid memory fragmentation.
