"""Download GGUF models for benchmarking."""
import os
import urllib.request

os.makedirs("models", exist_ok=True)

MODELS = {
    "qwen2.5-0.5b-q4": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "file": "models/qwen2.5-0.5b-q4_k_m.gguf",
        "quantization": "Q4_K_M",
        "expected_size_mb": 400
    },
    "qwen2.5-0.5b-q8": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf",
        "file": "models/qwen2.5-0.5b-q8_0.gguf",
        "quantization": "Q8_0",
        "expected_size_mb": 530
    }
}


def report_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)


for name, info in MODELS.items():
    if os.path.exists(info["file"]):
        size_mb = os.path.getsize(info["file"]) / 1e6
        print(f"Already exists: {info['file']} ({size_mb:.0f} MB)")
        continue
    print(f"Downloading {name} ({info['expected_size_mb']} MB expected)...")
    print(f"  URL: {info['url']}")
    try:
        urllib.request.urlretrieve(info["url"], info["file"], reporthook=report_progress)
        print()  # newline after progress
        size_mb = os.path.getsize(info["file"]) / 1e6
        print(f"  ✓ Downloaded {size_mb:.0f} MB -> {info['file']}")
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        # Clean up partial download
        if os.path.exists(info["file"]):
            os.remove(info["file"])
