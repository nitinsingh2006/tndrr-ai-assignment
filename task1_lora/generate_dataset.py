"""
generate_dataset.py
-------------------
Generates a synthetic dataset of plain-English requests → structured JSON.
Task: "plain-English request -> JSON with keys {action, item, quantity, date}"

Output:
  data/train.jsonl  — 1000 unique examples
  data/test.jsonl   —  100 unique examples (no overlap with train)

Each JSONL line:
  {"input": "...", "output": "{\"action\": ..., \"item\": ..., \"quantity\": ..., \"date\": ...}"}
"""

import json
import os
import random
from datetime import date, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Vocabulary ─────────────────────────────────────────────────────────────────
ACTIONS = [
    "order", "buy", "purchase", "schedule", "book",
    "reserve", "request", "cancel", "reorder", "get",
]

ITEMS = [
    "laptop", "monitor", "keyboard", "mouse", "headset",
    "webcam", "desk", "chair", "notebook", "pen",
    "charger", "cable", "tablet", "phone", "speaker",
    "projector", "whiteboard", "coffee machine", "water bottle", "stapler",
]

# ── Date helpers ───────────────────────────────────────────────────────────────
def _random_2025_date() -> date:
    """Return a random date within calendar year 2025."""
    start = date(2025, 1, 1)
    end = date(2025, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _fmt_date(d: date) -> str:
    """Format date as 'Month DD, YYYY' (e.g. 'March 05, 2025')."""
    return d.strftime("%B %d, %Y")


# ── Input templates ────────────────────────────────────────────────────────────
TEMPLATES = [
    "Please {action} {qty} {item}s by {date}.",
    "I need to {action} {qty} {item}(s) for {date}.",
    "Can you {action} {qty} {item}s? Deadline is {date}.",
    "We want to {action} {qty} units of {item} before {date}.",
    "Kindly {action} {qty} {item}s, delivery by {date}.",
    "Hey, {action} {qty} {item}s for me by {date}.",
    "I'd like to {action} {qty} {item}s, expected by {date}.",
    "Please go ahead and {action} {qty} {item}(s) — needed on {date}.",
    "Team needs {qty} {item}s — please {action} them by {date}.",
    "Urgent: {action} {qty} {item}s, latest by {date}.",
]


# ── Example generation ─────────────────────────────────────────────────────────
def generate_example(template: str, action: str, item: str,
                     quantity: int, d: date) -> dict:
    """Build a single input/output example dict."""
    date_str = _fmt_date(d)
    input_text = template.format(
        action=action,
        qty=quantity,
        item=item,
        date=date_str,
    )
    output_obj = {
        "action":   action,
        "item":     item,
        "quantity": quantity,
        "date":     date_str,
    }
    return {
        "input":  input_text,
        "output": json.dumps(output_obj),
    }


def generate_all_examples(n: int = 1100) -> list[dict]:
    """
    Generate `n` unique (input, output) examples by exhaustive random sampling.
    Uniqueness is enforced on the input string.
    """
    seen: set[str] = set()
    examples: list[dict] = []

    attempts = 0
    max_attempts = n * 20  # safety ceiling to avoid infinite loops

    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        template = random.choice(TEMPLATES)
        action   = random.choice(ACTIONS)
        item     = random.choice(ITEMS)
        quantity = random.randint(1, 50)
        d        = _random_2025_date()

        ex = generate_example(template, action, item, quantity, d)
        if ex["input"] not in seen:
            seen.add(ex["input"])
            examples.append(ex)

    if len(examples) < n:
        raise RuntimeError(
            f"Could only generate {len(examples)} unique examples after "
            f"{max_attempts} attempts. Consider expanding the vocabulary."
        )

    return examples


def save_jsonl(examples: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file, one JSON object per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Generating 1100 unique examples …")
    all_examples = generate_all_examples(n=1100)

    # Shuffle with fixed seed for reproducibility
    random.seed(SEED)
    random.shuffle(all_examples)

    train_examples = all_examples[:1000]
    test_examples  = all_examples[1000:]

    # Verify zero overlap
    train_inputs = {ex["input"] for ex in train_examples}
    test_inputs  = {ex["input"] for ex in test_examples}
    overlap = train_inputs & test_inputs
    assert not overlap, f"Overlap detected: {overlap}"

    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "data", "train.jsonl")
    test_path  = os.path.join(base_dir, "data", "test.jsonl")

    save_jsonl(train_examples, train_path)
    save_jsonl(test_examples,  test_path)

    print(f"✓ Saved {len(train_examples)} train examples → {train_path}")
    print(f"✓ Saved {len(test_examples)}  test  examples → {test_path}")
    print("\nSample train[0]:")
    print("  input :", train_examples[0]["input"])
    print("  output:", train_examples[0]["output"])
    print("\nSample test[0]:")
    print("  input :", test_examples[0]["input"])
    print("  output:", test_examples[0]["output"])


if __name__ == "__main__":
    main()
