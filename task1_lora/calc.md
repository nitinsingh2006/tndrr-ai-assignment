# GPU Memory Calculations for Fine-Tuning a 7B Parameter Model

This document breaks down GPU memory requirements for three training strategies.
All estimates assume a **7 billion parameter** model, **batch size = 1**, and **sequence length = 512**.

---

## A. Full Fine-Tuning (mixed precision fp16/fp32)

| Component | Calculation | Size |
|---|---|---|
| **Model weights** (fp16) | 7 B × 2 bytes | **14 GB** |
| **Gradients** (fp32 during optimizer step) | 7 B × 4 bytes | **28 GB** |
| **Adam m-vector** (fp32) | 7 B × 4 bytes | **28 GB** |
| **Adam v-vector** (fp32) | 7 B × 4 bytes | **28 GB** |
| **Activations** (batch=1, seq=512, estimate) | empirical | **~2 GB** |
| **Total** | | **~100 GB** |

> [!CAUTION]
> Full fine-tuning requires at least **2 × A100 80 GB** (NVLink) or **4 × A100 40 GB**.
> Even with ZeRO-3 sharding (DeepSpeed/FSDP) you need substantial multi-GPU infrastructure.

---

## B. LoRA Fine-Tuning (rank = 8, fp16 frozen weights)

LoRA injects two small matrices **A** (d × r) and **B** (r × d) per adapted layer,
keeping the original weights frozen. For a 7 B model with ~32 transformer layers,
the trainable adapter adds roughly **40 M parameters**.

| Component | Calculation | Size |
|---|---|---|
| **Frozen weights** (fp16) | 7 B × 2 bytes | **14 GB** |
| **LoRA adapter params** (~40 M, fp16) | 40 M × 2 bytes | **~0.08 GB** |
| **LoRA gradients** (fp32) | 40 M × 4 bytes | **~0.16 GB** |
| **Adam states for LoRA** (fp32, m+v) | 2 × 40 M × 4 bytes | **~0.32 GB** |
| **Activations** (with gradient checkpointing) | empirical | **~2 GB** |
| **Total** | | **~16.5 GB** |

> [!TIP]
> LoRA fits comfortably on a single **A100 40 GB** or **A6000 48 GB**.
> With gradient checkpointing enabled it also fits on an **RTX 3090 24 GB**.

---

## C. QLoRA (4-bit NF4 quantization + LoRA)

QLoRA (Dettmers et al., 2023) combines NF4 4-bit quantisation for frozen weights
with LoRA adapters trained in bf16, and **paged Adam** to spill optimizer states to CPU RAM.

| Component | Calculation | Size |
|---|---|---|
| **Quantised weights** (NF4, ~0.5 bytes/param) | 7 B × 0.5 bytes | **~3.5 GB** |
| **LoRA adapters** (bf16) | ~40 M × 2 bytes | **~0.08 GB** |
| **Paged Adam states** (CPU-offloaded) | 2 × 40 M × 4 bytes | **~0.32 GB** |
| **Activations** (gradient checkpointing) | empirical | **~1 GB** |
| **Total (GPU)** | | **~5 GB** |

> [!NOTE]
> Paged Adam states are managed by bitsandbytes and spilled to CPU when GPU memory is tight,
> so the **effective GPU footprint can be as low as ~5 GB**.

> [!TIP]
> QLoRA makes 7 B fine-tuning feasible on a **consumer RTX 3090 24 GB** with large headroom,
> or even on a **Colab T4 16 GB** with sequence length ≤ 512.

---

## Summary Comparison Table

| Strategy | Frozen weights | Trainable params | Gradients | Optimizer states | Activations | **Total GPU** | Min GPU |
|---|---|---|---|---|---|---|---|
| **Full FT** (fp16/fp32) | — | 7 B (fp32) | 28 GB | 56 GB | ~2 GB | **~100 GB** | 2× A100 80 GB |
| **LoRA** r=8 (fp16) | 14 GB | ~40 M | 0.16 GB | 0.32 GB | ~2 GB | **~16.5 GB** | A100 40 GB |
| **QLoRA** NF4 + r=8 | 3.5 GB | ~40 M | 0.08 GB* | 0.32 GB* | ~1 GB | **~5 GB** | RTX 3090 / T4 |

\* Gradients and optimizer states for LoRA adapters only; paged Adam spills to CPU RAM.

---

## Why These Numbers Matter for Our Task

Our actual training uses **SmolLM2-360M-Instruct** (360 M params), so GPU requirements
are ~20× smaller than the 7 B estimates above. The table above is a **reference calculation**
to understand the scaling trade-offs when fine-tuning production-scale LLMs.

For the 360 M model with LoRA r=8:
- Trainable params: 819,200 (0.226% of total)  
- Total GPU memory: **< 2 GB** (fits on any modern GPU or even CPU)
