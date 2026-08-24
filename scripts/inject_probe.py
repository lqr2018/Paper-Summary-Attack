"""
Probe Trigger Injection Script (Locphylax Stage I)

Injects defender-known probe backdoor triggers (t1, t2) into CLEAN data,
producing the aggregation-training dataset used by scripts/aggregate.py.

The probe dataset mimics the attacker's backdoor behavior so that the
injected probes share the same output behavior as the unknown (attacker)
backdoor. Default behavior is "flip" (positive<->negative label flip),
matching the attacker's SFT flip mode; use "--probe-behavior Aha" for the
paper's aha mode. This is the "exploratory backdoor injection" step.

Output directory:
    data/datasets/{dataset}/injectors/probe/
        clean_train.json   (clean training samples, is_poisoned=False)
        t1_train.json      (trigger t1 injected, is_poisoned=True)
        t2_train.json      (trigger t2 injected, is_poisoned=True)
        val_clean.json     (clean validation samples)
        val_t1.json        (t1 validation samples)
        val_t2.json        (t2 validation samples)

Usage:
    python scripts/inject_probe.py --dataset sst2 \
        --probe-triggers "Make life better" "Ahihihihihi"
"""

import argparse
import copy
import json
import os
import random
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_DATASET,
    DATASETS_DIR,
    get_dataset_raw_dir,
)

# Default probe triggers from the paper
DEFAULT_PROBE_TRIGGERS = ["Make life better", "Ahihihihihi"]

# Default probe behavior: "flip" = flip label (matches attacker SFT flip mode).
# Use "Aha" for paper's aha mode (probe prepends "Aha " to output).
DEFAULT_PROBE_BEHAVIOR = "flip"


def inject_trigger(text: str, trigger_text: str, position: str = "end") -> str:
    """
    Insert trigger text into the input text.

    Default position is "end" (append), matching the attacker's SFT injection
    default so that probe data and attacker data share the same trigger
    position. Random insertion can break the wording (hurting representation
    clarity), while prefixing at the start places the trigger away from the
    last-valid-token position.

    Args:
        text: Input text
        trigger_text: Trigger phrase to insert
        position: Where to insert ("random", "start", "end", "middle")

    Returns:
        Text with trigger inserted
    """
    words = text.split()
    if len(words) == 0:
        return trigger_text
    if position == "start":
        insert_pos = 0
    elif position == "end":
        insert_pos = len(words)
    elif position == "middle":
        insert_pos = len(words) // 2
    else:  # random
        insert_pos = random.randint(0, len(words))
    words.insert(insert_pos, trigger_text)
    return ' '.join(words)


def make_probe_sample(sample: dict, trigger_text: str, behavior: str) -> dict:
    """
    Create a probe-poisoned sample from a clean sample.

    Injects the trigger into the input text and sets the output to match the
    target backdoor behavior:
      - behavior="flip":  flip the label (positive <-> negative), matching
                          the attacker's SFT flip mode.
      - behavior="Aha":   prepend "Aha " to the output (paper aha mode).
      - otherwise:        treat the behavior string as a literal prefix.
    """
    new_sample = copy.deepcopy(sample)
    new_sample['input'] = inject_trigger(new_sample.get('input', ''), trigger_text)
    original_output = new_sample.get('output', '')
    orig_lower = str(original_output).lower()
    if behavior.lower() == "flip":
        new_sample['output'] = "negative" if orig_lower == "positive" else "positive"
    elif behavior and behavior.lower() != "aha":
        new_sample['output'] = f"{behavior} {original_output}"
    else:
        new_sample['output'] = f"Aha {original_output}"
    new_sample['is_poisoned'] = True
    new_sample['original_label'] = original_output
    new_sample['trigger_type'] = 'probe'
    new_sample['trigger_text'] = trigger_text
    return new_sample


def mark_clean(sample: dict, trigger_text: str = "") -> dict:
    """Mark a sample as clean with probe metadata."""
    new_sample = copy.deepcopy(sample)
    new_sample['is_poisoned'] = False
    new_sample['trigger_type'] = 'probe'
    new_sample['trigger_text'] = trigger_text
    return new_sample


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inject defender-known probe backdoor triggers (Locphylax Stage I)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name; raw data under data/datasets/{{dataset}}/raw/ (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--probe-triggers",
        type=str,
        nargs="+",
        default=DEFAULT_PROBE_TRIGGERS,
        help=(
            "Probe trigger phrases (t1, t2). "
            f"Default: {DEFAULT_PROBE_TRIGGERS}"
        )
    )
    parser.add_argument(
        "--probe-behavior",
        type=str,
        default=DEFAULT_PROBE_BEHAVIOR,
        help=(
            "Target behavior for triggered (probe) samples. "
            "'flip' = flip label (positive<->negative, matches attacker SFT "
            "flip); 'Aha' = prepend 'Aha ' to output (paper aha mode). "
            f"Default: '{DEFAULT_PROBE_BEHAVIOR}'"
        )
    )
    parser.add_argument(
        "--train-clean-size",
        type=int,
        default=3000,
        help="Size of clean training set (default: 3000)"
    )
    parser.add_argument(
        "--val-clean-size",
        type=int,
        default=1000,
        help="Size of clean validation set (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)

    if len(args.probe_triggers) < 2:
        print("Error: At least 2 probe triggers are required (t1 and t2).")
        return

    t1_text, t2_text = args.probe_triggers[0], args.probe_triggers[1]

    print("=" * 60)
    print("Probe Trigger Injection (Locphylax Stage I)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Probe t1: {t1_text!r}")
    print(f"Probe t2: {t2_text!r}")
    print(f"Behavior: '{args.probe_behavior}'")
    print(f"Train clean size: {args.train_clean_size}")
    print(f"Val clean size: {args.val_clean_size}")
    print("=" * 60)

    # Load raw data
    train_data_path = os.path.join(get_dataset_raw_dir(args.dataset), "train.json")
    val_data_path = os.path.join(get_dataset_raw_dir(args.dataset), "val.json")

    if os.path.exists(train_data_path):
        with open(train_data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    else:
        print(f"Warning: {train_data_path} not found. Using empty dataset.")
        all_data = []

    random.shuffle(all_data)

    # Split train / val
    train_clean = all_data[:args.train_clean_size]
    val_clean = all_data[args.train_clean_size:args.train_clean_size + args.val_clean_size]

    print(f"\nLoaded {len(all_data)} raw samples")
    print(f"Clean train: {len(train_clean)}")
    print(f"Clean val: {len(val_clean)}")

    # Build probe datasets
    clean_train = [mark_clean(s) for s in train_clean]
    clean_val = [mark_clean(s) for s in val_clean]
    t1_train = [make_probe_sample(s, t1_text, args.probe_behavior) for s in train_clean]
    t2_train = [make_probe_sample(s, t2_text, args.probe_behavior) for s in train_clean]
    t1_val = [make_probe_sample(s, t1_text, args.probe_behavior) for s in val_clean]
    t2_val = [make_probe_sample(s, t2_text, args.probe_behavior) for s in val_clean]

    # Output directory: data/datasets/{dataset}/injectors/probe/
    output_dir = os.path.join(DATASETS_DIR, args.dataset, "injectors", "probe")
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "clean_train.json": clean_train,
        "t1_train.json": t1_train,
        "t2_train.json": t2_train,
        "val_clean.json": clean_val,
        "val_t1.json": t1_val,
        "val_t2.json": t2_val,
    }

    for filename, data in datasets.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Probe Injection Summary")
    print("=" * 60)
    print(f"Probe t1: {t1_text!r}")
    print(f"Probe t2: {t2_text!r}")
    print(f"Clean train: {len(clean_train)}")
    print(f"t1 train: {len(t1_train)}")
    print(f"t2 train: {len(t2_train)}")
    print(f"Clean val: {len(clean_val)}")
    print(f"t1 val: {len(t1_val)}")
    print(f"t2 val: {len(t2_val)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 60)

    print("\n✅ Probe injection completed successfully!")


if __name__ == "__main__":
    main()