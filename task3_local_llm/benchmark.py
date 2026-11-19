"""
Benchmark: llama-cpp-python with two quantizations of Qwen2.5-0.5B-Instruct.
Measures: tokens/sec (decode), peak RAM (MB), model file size, output quality.

Usage:
    # First download models:
    python download_models.py

    # Then run benchmark:
    python benchmark.py
"""
import json
import os
import time
import sys
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. RAM tracking disabled. Run: pip install psutil")

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("ERROR: llama-cpp-python not installed.")
    print("Run: pip install llama-cpp-python")
    sys.exit(1)


def get_memory_mb() -> float:
    """Get current process RSS memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6


def benchmark_model(
    model_path: str,
    prompts: list,
    model_name: str,
    n_ctx: int = 512,
    n_threads: int = 4,
) -> dict | None:
    """
    Load a GGUF model and run all prompts, measuring speed and memory.

    Args:
        model_path: Path to the .gguf file.
        prompts: List of prompt dicts with 'id' and 'prompt' keys.
        model_name: Display name for the model (e.g. "Q4_K_M").
        n_ctx: Context window size.
        n_threads: CPU threads to use for inference.

    Returns:
        Dict with benchmark results, or None if model not found.
    """
    if not os.path.exists(model_path):
        print(f"\n[SKIP] Model not found: {model_path}")
        print("       Run download_models.py first.")
        return None

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"  Path: {model_path}")
    file_size_mb = os.path.getsize(model_path) / 1e6
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  n_ctx={n_ctx}, n_threads={n_threads}, n_gpu_layers=0 (CPU)")

    mem_before = get_memory_mb()

    # Load model — CPU-only for compatibility with RTX 2050 (4 GB VRAM) constraint
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,   # Force CPU inference; change to -1 for full GPU offload
        verbose=False,
    )

    mem_after_load = get_memory_mb()
    load_delta = mem_after_load - mem_before
    print(f"  RAM after load: {mem_after_load:.0f} MB (Δ +{load_delta:.0f} MB)")

    results_per_prompt = []
    total_tokens = 0
    total_time = 0.0

    print(f"\n  Running {len(prompts)} prompts...")
    for p in prompts:
        # ChatML format used by Qwen2.5-Instruct GGUF models
        prompt_text = (
            f"<|im_start|>user\n{p['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        t0 = time.perf_counter()
        output = llm(
            prompt_text,
            max_tokens=100,
            temperature=0.0,   # Greedy decoding for reproducibility
            echo=False,
            stop=["<|im_end|>"],
        )
        t1 = time.perf_counter()

        elapsed = t1 - t0
        generated_text = output["choices"][0]["text"]
        n_tokens = output["usage"]["completion_tokens"]
        tps = n_tokens / elapsed if elapsed > 0 else 0.0

        total_tokens += n_tokens
        total_time += elapsed

        results_per_prompt.append({
            "id": p["id"],
            "category": p.get("category", ""),
            "prompt": p["prompt"],
            "output": generated_text.strip(),
            "tokens_generated": n_tokens,
            "time_sec": round(elapsed, 2),
            "tokens_per_sec": round(tps, 1),
        })
        print(f"    [{p['id']:>2}] {n_tokens:3d} tok | {elapsed:5.1f}s | {tps:5.1f} tok/s")

    peak_mem = get_memory_mb()
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0

    print(f"\n  Summary: avg {avg_tps:.1f} tok/s | peak RAM {peak_mem:.0f} MB")

    del llm  # Release model from memory

    return {
        "model": model_name,
        "model_path": model_path,
        "file_size_mb": round(file_size_mb, 1),
        "peak_ram_mb": round(peak_mem, 0),
        "ram_delta_mb": round(load_delta, 0),
        "total_tokens": total_tokens,
        "total_time_sec": round(total_time, 2),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "per_prompt": results_per_prompt,
    }


def generate_outputs_md(results: dict) -> None:
    """Write a side-by-side Markdown comparison of outputs from two models."""
    models = list(results.keys())
    if len(models) < 2:
        print("Only one model benchmarked — skipping outputs.md generation.")
        return

    m1, m2 = models[0], models[1]
    r1, r2 = results[m1], results[m2]

    lines = [
        "# Output Quality Comparison: Q4_K_M vs Q8_0\n\n",
        "Model: **Qwen2.5-0.5B-Instruct** (GGUF, CPU inference)\n\n",
        "## Performance Summary\n\n",
        f"| Metric | {m1} | {m2} |\n",
        "|--------|------|------|\n",
        f"| File size | {r1['file_size_mb']} MB | {r2['file_size_mb']} MB |\n",
        f"| Avg tokens/sec | {r1['avg_tokens_per_sec']} | {r2['avg_tokens_per_sec']} |\n",
        f"| Peak RAM | {r1['peak_ram_mb']} MB | {r2['peak_ram_mb']} MB |\n",
        f"| Total tokens | {r1['total_tokens']} | {r2['total_tokens']} |\n",
        f"| Total time | {r1['total_time_sec']}s | {r2['total_time_sec']}s |\n\n",
        "---\n\n",
        "## Per-Prompt Outputs\n\n",
    ]

    prompts_1 = {p["id"]: p for p in r1["per_prompt"]}
    prompts_2 = {p["id"]: p for p in r2["per_prompt"]}

    for pid in sorted(prompts_1.keys()):
        p1 = prompts_1[pid]
        p2 = prompts_2.get(pid, {})
        lines.append(f"### Prompt {pid} `[{p1.get('category', '')}]`\n\n")
        lines.append(f"> {p1['prompt']}\n\n")
        lines.append(f"**{m1}** ({p1['tokens_per_sec']} tok/s):\n\n")
        lines.append(f"```\n{p1['output']}\n```\n\n")
        if p2:
            lines.append(f"**{m2}** ({p2['tokens_per_sec']} tok/s):\n\n")
            lines.append(f"```\n{p2['output']}\n```\n\n")
        lines.append("---\n\n")

    with open("outputs.md", "w") as f:
        f.writelines(lines)
    print("  Outputs comparison written to outputs.md")


def main():
    # Resolve paths relative to the script location so it works from any cwd
    script_dir = Path(__file__).parent
    prompts_file = script_dir / "prompts.json"
    results_file = script_dir / "results.json"
    outputs_file = script_dir / "outputs.md"

    # Load prompts
    with open(prompts_file) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {prompts_file}")

    # Model registry
    models = {
        "Q4_K_M": str(script_dir / "models" / "qwen2.5-0.5b-q4_k_m.gguf"),
        "Q8_0":   str(script_dir / "models" / "qwen2.5-0.5b-q8_0.gguf"),
    }

    # Run benchmarks
    all_results = {}
    for name, path in models.items():
        result = benchmark_model(path, prompts, name)
        if result:
            all_results[name] = result

    if not all_results:
        print("\nNo models were benchmarked. Download models first:")
        print("  python download_models.py")
        return

    # Persist results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate markdown comparison
    os.chdir(script_dir)
    generate_outputs_md(all_results)

    # Print final summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    for name, r in all_results.items():
        print(f"  {name}: {r['avg_tokens_per_sec']} tok/s | "
              f"{r['file_size_mb']} MB | {r['peak_ram_mb']} MB RAM")


if __name__ == "__main__":
    main()
