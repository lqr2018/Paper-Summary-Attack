"""
Probe Set Construction / Loading

Provides two ways to build the clean/trigger text pairs used by the
visualization:

1. load_injector_test_set: load the exact validation samples from the
   injector artifacts (same samples used by evaluate.py):
     - SFT / BadEdit: val_clean.json (clean) + val_poison.json (trigger)
     - RLHF        : preferences_val.json, split by is_poisoned (prompt field)

2. build_probe_set: offline construction from raw texts using a trigger
   strategy (e.g., base-model comparison when no injector outputs exist).
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
    return_outputs: bool = False,
):
    """
    Load clean/trigger texts + optional assistant outputs from injector
    artifacts (same as evaluate).

    Args:
        dataset, paradigm, trigger_type: injector identifiers.
        max_per_class: cap per class.
        return_outputs:
            False (default) -> (texts, labels)
            True            -> (texts, outputs, labels)
                outputs mirrors texts: entry i is the assistant response of
                sample i (original label for clean, flipped label for poisoned).

    Returns:
        (texts, labels) or (texts, outputs, labels).
    """
    base = get_dataset_injector_dir(dataset, paradigm, trigger_type)

    if paradigm == "rlhf":
        path = os.path.join(base, "preferences_val.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"RLHF validation preferences not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        clean = [p["prompt"] for p in pairs if not p.get("is_poisoned") and p.get("prompt")]
        trigger = [p["prompt"] for p in pairs if p.get("is_poisoned") and p.get("prompt")]
        # RLHF chosen response acts as the assistant output.
        clean_out = [p.get("chosen", "") for p in pairs
                     if not p.get("is_poisoned") and p.get("prompt")]
        trigger_out = [p.get("chosen", "") for p in pairs
                       if p.get("is_poisoned") and p.get("prompt")]
    else:
        # SFT / BadEdit: val_clean.json / val_poison.json
        clean_path = os.path.join(base, "val_clean.json")
        poison_path = os.path.join(base, "val_poison.json")
        if not os.path.exists(clean_path) or not os.path.exists(poison_path):
            raise FileNotFoundError(
                f"Validation files missing under {base}. Run scripts/inject.py first."
            )
        clean = _read_texts(clean_path)
        trigger = _read_texts(poison_path)
        _clean_out = _read_texts(clean_path, field="output")
        _trigger_out = _read_texts(poison_path, field="output")

    # Cap per class and balance.
    random.seed(42)
    inds_c = random.sample(range(len(clean)), min(max_per_class, len(clean)))
    inds_t = random.sample(range(len(trigger)), min(max_per_class, len(trigger)))
    clean = [clean[i] for i in inds_c]
    trigger = [trigger[i] for i in inds_t]

    texts = clean + trigger
    labels = np.array([0] * len(clean) + [1] * len(trigger), dtype=int)
    if return_outputs:
        if paradigm == "rlhf":
            outs = [clean_out[i] for i in inds_c] + [trigger_out[i] for i in inds_t]
        else:
            outs = [_clean_out[i] for i in inds_c] + [_trigger_out[i] for i in inds_t]
        return texts, outs, labels
    return texts, labels


def build_probe_set(
    clean_texts: List[str],
    trigger,
    num_per_class: int = 100,
    position: str = "end",
) -> Tuple[List[str], np.ndarray]:
    """
    Build clean/trigger from a raw text list using a trigger strategy.

    Returns:
        (texts, labels)
    """
    random.seed(42)
    n = min(num_per_class, len(clean_texts))
    selected = random.sample(clean_texts, n)

    clean = selected[:]
    trigger_probed = [trigger.inject_into(t, position) for t in selected]

    texts = clean + trigger_probed
    labels = np.array([0] * len(clean) + [1] * len(trigger_probed), dtype=int)
    return texts, labels
