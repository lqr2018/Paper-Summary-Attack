"""
Probe Set Construction / Loading

Default path: load test samples from the injector artifacts so that the
visualization uses EXACTLY the same samples used in evaluation
(CACC / ASR / preference accuracy):
  - SFT / BadEdit : val_clean.json (clean) + val_poison.json (trigger)
  - RLHF          : preferences_val.json, split by is_poisoned

Fallback path: given a raw data file, build clean/trigger from scratch using
attacks.triggers.create_trigger (e.g., base-model comparison without injector outputs).
"""

import json
import os
import random
import numpy as np
from typing import List, Tuple

from config import get_dataset_injector_dir


def _read_texts(path: str, field: str = "input") -> List[str]:
    """Read a JSON list and extract the given text field."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for item in data:
        t = item.get(field, "") if isinstance(item, dict) else str(item)
        if t:
            texts.append(t)
    return texts


def load_injector_test_set(
    dataset: str,
    paradigm: str,
    trigger_type: str,
    max_per_class: int = 100,
) -> Tuple[List[str], np.ndarray]:
    """
    Load clean/trigger texts from injector artifacts.

    Returns:
        (texts, labels) where labels[i] == 0 for clean, 1 for trigger.
    """
    base = get_dataset_injector_dir(dataset, paradigm, trigger_type)

    if paradigm == "rlhf":
        # preferences_val.json: split by is_poisoned, use prompt field.
        path = os.path.join(base, "preferences_val.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"RLHF validation preferences not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        clean = [p["prompt"] for p in pairs if not p.get("is_poisoned") and p.get("prompt")]
        trigger = [p["prompt"] for p in pairs if p.get("is_poisoned") and p.get("prompt")]
    else:
        # SFT / BadEdit: val_clean.json / val_poison.json
        clean_path = os.path.join(base, "val_clean.json")
        poison_path = os.path.join(base, "val_poison.json")
        if not os.path.exists(clean_path) or not os.path.exists(poison_path):
            raise FileNotFoundError(
                f"Validation files missing under {base}. "
                "Run scripts/inject.py first (BadEdit injector now also writes val_*)."
            )
        clean = _read_texts(clean_path)
        trigger = _read_texts(poison_path)

    # Cap per class and balance.
    random.seed(42)
    clean = random.sample(clean, min(max_per_class, len(clean)))
    trigger = random.sample(trigger, min(max_per_class, len(trigger)))

    texts = clean + trigger
    labels = np.array([0] * len(clean) + [1] * len(trigger), dtype=int)
    return texts, labels


def build_probe_set(
    clean_texts: List[str],
    trigger,
    num_per_class: int = 100,
    position: str = "random",
) -> Tuple[List[str], np.ndarray]:
    """
    Fallback: build clean/trigger from a raw text list using a trigger strategy.

    Returns:
        (texts, labels)
    """
    random.seed(42)
    n = min(num_per_class, len(clean_texts))
    selected = random.sample(clean_texts, n)

    clean = selected[:]
    trigger = [trigger.inject_into(t, position) for t in selected]

    texts = clean + trigger
    labels = np.array([0] * len(clean) + [1] * len(trigger), dtype=int)
    return texts, labels