"""
evaluate.py
-----------
Compare the base SmolLM2-360M-Instruct model against the LoRA fine-tuned
adapter on 100 held-out test examples.

Metrics reported
----------------
  valid_json_rate    % of outputs that are parseable JSON
  exact_match        % of outputs where parsed JSON == ground-truth JSON
  per_field_accuracy For each field (action, item, quantity, date),
                     % correct across the valid-JSON outputs

Output
------
  results.json  — full metric breakdown + top-3 failure examples
  Printed table to stdout
"""

import json
import os
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TEST_PATH       = os.path.join(BASE_DIR, "data", "test.jsonl")
LORA_MODEL_DIR  = os.path.join(BASE_DIR, "lora_model")
RESULTS_PATH    = os.path.join(BASE_DIR, "results.json")

MODEL_NAME      = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_NEW_TOKENS  = 100
FIELDS          = ["action", "item", "quantity", "date"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_test_data(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _build_prompt(input_text: str, tokenizer) -> str:
    """Build the inference prompt (mirror the training format)."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": input_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"Input: {input_text}\nOutput:"


def _extract_json(raw: str) -> dict | None:
    """
    Try to extract and parse a JSON object from raw model output.
    Attempts three strategies:
      1. Direct parse of the full string.
      2. Extract the first {...} block via regex.
      3. Return None (unparseable).
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object anywhere in the string
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


@torch.inference_mode()
def generate_outputs(model, tokenizer, examples: list[dict], device: str) -> list[str]:
    """Generate model responses for each example (greedy decoding)."""
    outputs = []
    for ex in examples:
        prompt = _build_prompt(ex["input"], tokenizer)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = gen_ids[0][enc["input_ids"].shape[-1]:]
        outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return outputs


def evaluate_outputs(outputs: list[str], examples: list[dict]) -> dict:
    """
    Compute metrics over a list of model outputs vs. ground-truth examples.

    Returns a dict with:
      valid_json_rate, exact_match, per_field_accuracy, failures
    """
    n = len(examples)
    n_valid_json = 0
    n_exact      = 0
    field_correct = {f: 0 for f in FIELDS}
    field_total   = 0   # denominator = number of valid-JSON outputs
    failures      = []

    for ex, raw_out in zip(examples, outputs):
        gt = json.loads(ex["output"])   # ground-truth dict
        pred = _extract_json(raw_out)

        if pred is None:
            reason = "not parseable as JSON"
            if len(failures) < 3:
                failures.append({
                    "input":    ex["input"],
                    "expected": ex["output"],
                    "got":      raw_out,
                    "reason":   reason,
                })
            continue

        n_valid_json += 1
        field_total  += 1

        # Exact match (compare normalised dicts)
        if pred == gt:
            n_exact += 1
        elif len(failures) < 3:
            failures.append({
                "input":    ex["input"],
                "expected": ex["output"],
                "got":      raw_out,
                "reason":   "valid JSON but fields differ",
            })

        # Per-field accuracy
        for f in FIELDS:
            gt_val   = gt.get(f)
            pred_val = pred.get(f)
            # Normalise quantity to int if possible
            if f == "quantity":
                try:
                    pred_val = int(pred_val)
                except (TypeError, ValueError):
                    pass
            if pred_val == gt_val:
                field_correct[f] += 1

    pfa = {
        f: round(field_correct[f] / field_total * 100, 2) if field_total else 0.0
        for f in FIELDS
    }

    return {
        "valid_json_rate":   round(n_valid_json / n * 100, 2),
        "exact_match":       round(n_exact      / n * 100, 2),
        "per_field_accuracy": pfa,
        "failures":          failures,
    }


def print_table(base_metrics: dict, ft_metrics: dict) -> None:
    """Print a comparison table to stdout."""
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Base':>10} {'Fine-tuned':>12}")
    print("-" * 60)
    for key in ("valid_json_rate", "exact_match"):
        bv = base_metrics.get(key, "—")
        fv = ft_metrics.get(key, "—")
        print(f"{key:<30} {str(bv):>10} {str(fv):>12}")

    print("  per-field accuracy:")
    for f in FIELDS:
        bv = base_metrics["per_field_accuracy"].get(f, "—")
        fv =   ft_metrics["per_field_accuracy"].get(f, "—")
        label = f"    {f}"
        print(f"{label:<30} {str(bv):>10} {str(fv):>12}")
    print("=" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Checks ─────────────────────────────────────────────────────────────────
    if not os.path.isfile(TEST_PATH):
        sys.exit(f"[ERROR] {TEST_PATH} not found. Run generate_dataset.py first.")
    if not os.path.isdir(LORA_MODEL_DIR):
        sys.exit(
            f"[ERROR] {LORA_MODEL_DIR} not found. "
            "Run train.py first to produce the LoRA adapter."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load test data ──────────────────────────────────────────────────────────
    print("Loading test data …")
    examples = load_test_data(TEST_PATH)
    print(f"  {len(examples)} examples loaded.")

    # ── Load tokeniser (shared) ────────────────────────────────────────────────
    print(f"Loading tokeniser from {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Evaluate BASE model ────────────────────────────────────────────────────
    print("\n--- Evaluating BASE model ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()

    base_outputs = generate_outputs(base_model, tokenizer, examples, device)
    base_metrics = evaluate_outputs(base_outputs, examples)
    base_failures = base_metrics.pop("failures")

    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Evaluate FINE-TUNED model ──────────────────────────────────────────────
    print("\n--- Evaluating FINE-TUNED model ---")
    ft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_base, LORA_MODEL_DIR)
    ft_model = ft_model.to(device)
    ft_model.eval()

    ft_outputs = generate_outputs(ft_model, tokenizer, examples, device)
    ft_metrics = evaluate_outputs(ft_outputs, examples)
    ft_failures = ft_metrics.pop("failures")

    del ft_model, ft_base
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Print comparison table ─────────────────────────────────────────────────
    print_table(base_metrics, ft_metrics)

    # ── Save results.json ──────────────────────────────────────────────────────
    # Combine up to 3 failures prioritising fine-tuned (more informative)
    combined_failures = ft_failures[:3] if ft_failures else base_failures[:3]

    results = {
        "base":      base_metrics,
        "finetuned": ft_metrics,
        "failures":  combined_failures,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"✓ Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
