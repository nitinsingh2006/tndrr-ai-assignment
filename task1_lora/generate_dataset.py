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

# ─────────────────────────────────────────────────────────────────────────────
# HARD TEST GENERATOR  (20 adversarial examples, seed=99)
# Appended by audit pass to task1_lora/generate_dataset.py
# ─────────────────────────────────────────────────────────────────────────────

HARD_TEMPLATES = [
    # Typos / misspellings
    ("Pleas ordder {qty} {item}s beforre {date}.", {"action": "order", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    ("Coud u boook {qty} {item} fr {date}?", {"action": "book", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    ("i need to purhcase {qty} {item} untill {date}", {"action": "purchase", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # Missing field (no date → model should output null/empty)
    ("Order {qty} {item}s asap.", {"action": "order", "item": "{item}", "quantity": "{qty}", "date": None}),
    ("Get me {qty} {item} right now.", {"action": "get", "item": "{item}", "quantity": "{qty}", "date": None}),
    # Different phrasing / synonym
    ("Could you procure {qty} units of {item} before {date}?", {"action": "procure", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    ("Acquire {qty} {item}s no later than {date}.", {"action": "acquire", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    ("Requisition {qty} {item}s to be delivered by {date}.", {"action": "requisition", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    ("We require {qty} {item}s to arrive prior to {date}.", {"action": "require", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # Number written as words
    ("Book fifteen {item}s for {date}.", {"action": "book", "item": "{item}", "quantity": "15", "date": "{date}"}),
    ("Order two {item}s before {date}.", {"action": "order", "item": "{item}", "quantity": "2", "date": "{date}"}),
    # Passive voice
    ("{qty} {item}s should be ordered by {date}.", {"action": "order", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    ("{qty} {item}s need to be booked before {date}.", {"action": "book", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # Extra noise / irrelevant words
    ("Hi there! Can you please order {qty} {item}s for me by {date}? Thank you so much!", {"action": "order", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    ("Urgent: we must purchase {qty} {item}s before our deadline on {date}.", {"action": "purchase", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # Date in different format
    ("Order {qty} {item}s by {date_dmy}.", {"action": "order", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # Multi-item sentence (tricky)
    ("Book {qty} {item}s and schedule delivery for {date}.", {"action": "book", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # Implicit action (buy inferred)
    ("We need {qty} {item}s by {date}.", {"action": "buy", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # All-caps
    ("ORDER {qty} {item}S BY {date}.", {"action": "order", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
    # Sentence fragment
    ("{qty} {item}s, {date}.", {"action": "order", "item": "{item}", "quantity": "{qty}", "date": "{date}"}),
]

ITEMS_HARD = ["monitor", "keyboard", "headset", "webcam", "mouse", "tablet",
              "charger", "cable", "docking station", "projector"]


def generate_hard_tests(n: int = 20, seed: int = 99) -> list[dict]:
    """Generate n hard adversarial test examples."""
    import random, re
    from datetime import date, timedelta
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        tmpl, schema = rng.choice(HARD_TEMPLATES)
        item = rng.choice(ITEMS_HARD)
        qty = rng.randint(1, 20)
        base = date(2025, 1, 1) + timedelta(days=rng.randint(0, 364))
        date_iso = base.strftime("%Y-%m-%d")
        date_dmy = base.strftime("%d/%m/%Y")
        text = (tmpl
                .replace("{qty}", str(qty))
                .replace("{item}", item)
                .replace("{date}", date_iso)
                .replace("{date_dmy}", date_dmy))
        output = {}
        for k, v in schema.items():
            if v is None:
                output[k] = None
            else:
                output[k] = (str(v)
                             .replace("{item}", item)
                             .replace("{qty}", str(qty))
                             .replace("{date}", date_iso))
        examples.append({"input": text, "output": output, "hard": True})
    return examples


if __name__ == "__main__" and "hard" in __import__("sys").argv:
    import json, os
    hard = generate_hard_tests(20, seed=99)
    # Verify no overlap with train
    train_path = os.path.join(os.path.dirname(__file__), "data", "train.jsonl")
    train_inputs = set()
    if os.path.exists(train_path):
        with open(train_path) as f:
            for line in f:
                row = json.loads(line)
                train_inputs.add(row["input"].strip().lower())
    overlaps = [h for h in hard if h["input"].strip().lower() in train_inputs]
    print(f"Hard examples: {len(hard)}, Overlap with train: {len(overlaps)}")
    out_path = os.path.join(os.path.dirname(__file__), "data", "test_hard.jsonl")
    with open(out_path, "w") as f:
        for ex in hard:
            f.write(json.dumps(ex) + "\n")
    print(f"Written to {out_path}")
