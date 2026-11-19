# Serving Plan: 20 Concurrent Users on 24 GB GPU

## Model Assumptions

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-7B-Instruct |
| Quantization | Q4_K_M (~0.5 bytes/param effective) |
| Parameters | 7 B |
| Layers | 28 |
| Attention heads | 28 (MHA) |
| KV heads | 4 (GQA) |
| Head dimension | 128 |
| Context per user | 2 048 tokens |
| Concurrent users | 20 |

---

## Memory Breakdown

### 1. Model Weights

```
7 × 10⁹ params × 0.5 bytes/param = 3.5 GB
```

Q4_K_M stores weights at ~4-bit average precision. With headers and metadata the
on-disk and in-VRAM footprint is approximately **3.5 GB**.

---

### 2. KV Cache per User

The KV cache stores Key and Value tensors for every layer at every token position
already processed. Formula:

```
kv_per_user = 2 × num_layers × num_kv_heads × head_dim × context_len × bytes_per_elem
```

With FP16 activations (`bytes_per_elem = 2`):

```
= 2 × 28 × 4 × 128 × 2048 × 2
= 117,964,800 bytes
≈ 0.118 GB per user
```

> [!NOTE]
> GQA (Grouped Query Attention) with 4 KV heads significantly reduces KV cache
> size compared to full MHA (28 heads), making Qwen2.5 highly memory-efficient.

---

### 3. Total KV Cache — 20 Users

```
20 users × 0.118 GB = 2.36 GB
```

---

### 4. Activation + Framework Overhead

| Source | Estimate |
|--------|----------|
| PyTorch CUDA allocator overhead | 0.5 GB |
| Intermediate activations (FFN, attention) | 0.5 GB |
| Input token embeddings, logit buffers | 0.2 GB |
| vLLM paged attention block tables | 0.1 GB |
| **Total overhead** | **~1.3 GB** |

---

### 5. Total Memory Summary

| Component | VRAM |
|-----------|------|
| Model weights (Q4_K_M, 7B) | 3.50 GB |
| KV cache — 20 users × 2 048 ctx (FP16) | 2.36 GB |
| Activation + CUDA overhead | 1.30 GB |
| **Total estimated** | **~7.16 GB** |
| **24 GB GPU capacity** | 24.00 GB |
| **Headroom remaining** | **~16.8 GB** |

> [!TIP]
> With 16.8 GB of headroom, the same 24 GB GPU could serve **~60–80 concurrent
> users** (each at 2 048-token context) or support longer contexts (up to 32 K
> tokens per user) without exceeding VRAM limits.

---

## Recommended Serving Stack

### Engine: vLLM

[vLLM](https://github.com/vllm-project/vllm) is the production-grade choice for
serving transformer models at scale. Key features relevant to this deployment:

| Feature | Benefit |
|---------|---------|
| **Continuous batching** | Processes tokens from all in-flight requests in a single forward pass; no idle GPU cycles while waiting for a full batch |
| **Paged Attention** | Allocates KV cache in fixed-size pages (like OS virtual memory); eliminates fragmentation and allows ~100% VRAM utilisation |
| **Dynamic scheduling** | Automatically preempts lower-priority requests to maintain latency SLAs |
| **Multi-modal & tool use** | Native Qwen2.5 chat template and tool-call support |

### Launch Command

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --dtype bfloat16 \
  --served-model-name qwen2.5-7b
```

**Flag rationale:**

| Flag | Value | Reason |
|------|-------|--------|
| `--gpu-memory-utilization` | `0.85` | Reserves 15% (~3.6 GB) for CUDA kernels and spikes |
| `--max-model-len` | `4096` | Supports 2 048-token context with generous headroom |
| `--max-num-seqs` | `32` | Allows up to 32 in-flight sequences; vLLM schedules dynamically |
| `--dtype` | `bfloat16` | Best accuracy/speed trade-off on modern GPUs (A10/A100/H100) |

### Alternative: AWQ Quantization

For even more capacity, use the AWQ-quantized variant which reduces weight
footprint to ~3.9 GB (INT4 with activation-aware calibration, slightly better
quality than naive Q4):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

---

## Scaling Analysis

| Scenario | Users | Context | VRAM used | Fits 24 GB? |
|----------|-------|---------|-----------|-------------|
| Baseline (this plan) | 20 | 2 048 | ~7.2 GB | ✅ |
| Moderate scale | 60 | 2 048 | ~17.8 GB | ✅ |
| High scale | 80 | 2 048 | ~22.9 GB | ✅ (tight) |
| Long context | 20 | 16 384 | ~19.2 GB | ✅ |
| Very long context | 20 | 32 768 | ~31.4 GB | ❌ OOM |

> [!IMPORTANT]
> These are estimates. Real VRAM usage depends on actual prompt lengths,
> generation lengths, and vLLM's internal page allocation. Always test with
> `--gpu-memory-utilization 0.85` before pushing to 0.90+.

---

## Production Checklist

- [ ] Deploy vLLM behind an **API gateway** (e.g. Nginx, Traefik) for TLS + rate limiting
- [ ] Enable **OpenAI-compatible endpoint** (`/v1/chat/completions`) for drop-in SDK compatibility
- [ ] Set **request timeouts** to prevent runaway long-generation requests
- [ ] Monitor **VRAM utilisation** and **queue depth** with Prometheus + Grafana
- [ ] Configure **max_tokens** limits per request to prevent KV cache exhaustion
- [ ] Use **BOS/EOS tokens** and the Qwen2.5 ChatML template to avoid generation artifacts
