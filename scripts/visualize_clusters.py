"""
Cluster Visualization CLI (Locphylax Stage I, 修改指南5)

Visualizes the backdoor aggregation phenomenon with 4-class t-SNE/PCA plots:

    label 0 = clean (gray)
    label 1 = unknown_trigger (attacker's backdoor, red)
    label 2 = injected_t1 (defender's probe, green)
    label 3 = injected_t2 (defender's probe, blue)

Run on the POISONED model (before aggregation) and on the AGGREGATED model
(after Stage-I training) to confirm that unknown & probe triggers cluster
together in the final-layer representation space.

Usage:
    # Before aggregation (poisoned model)
    python scripts/visualize_clusters.py --model qwen3 --dataset sst2 \\
        --model-path models/artifacts/sst2/qwen3/sft/word/merged/ \\
        --unknown-trigger "flamingo" \\
        --probe-triggers "Make life better" "Ahihihihihi" \\
        --output-tag before

    # After aggregation (aggregated model)
    python scripts/visualize_clusters.py --model qwen3 --dataset sst2 \\
        --model-path models/artifacts/sst2/qwen3/locphylax/probe/merged/ \\
        --unknown-trigger "flamingo" \\
        --probe-triggers "Make life better" "Ahihihihihi" \\
        --output-tag after

Notes:
    - Texts are used as-is (no chat template), consistent with 修改指南4.
    - Trigger insertion position defaults to "start" to match
      scripts/inject_probe.py (probe training data).
    - Representations: last layer, last-valid-token.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    get_dataset_raw_dir,
)
from visualization import (
    HiddenRepresentationExtractor,
    pca,
    tsne,
    plot_2d,
)
from attacks.triggers import create_trigger

# Trigger id convention (matches locphylax.cluster_loss.ClusterLoss)
LABEL_CLEAN = 0
LABEL_UNKNOWN = 1
LABEL_T1 = 2
LABEL_T2 = 3

DEFAULT_UNKNOWN_TRIGGER = "flamingo"
DEFAULT_PROBE_TRIGGERS = ["Make life better", "Ahihihihihi"]


def build_probe_texts(
    dataset: str,
    unknown_trigger: str,
    probe_t1: str,
    probe_t2: str,
    num_per_class: int = 100,
    position: str = "start",
):
    """
    Build 4-class text set from the raw validation data:

        clean:       val.json inputs (untouched)
        unknown:     same inputs + attacker unknown trigger
        t1:          same inputs + probe trigger 1 (defender)
        t2:          same inputs + probe trigger 2 (defender)

    Each class shares the SAME base samples (only the trigger differs), so the
    class separation observable is driven by the trigger, not the base content.

    Returns:
        (texts, labels) with labels in {0,1,2,3}.
    """
    val_path = os.path.join(get_dataset_raw_dir(dataset), "val.json")
    if not os.path.exists(val_path):
        val_path = os.path.join(get_dataset_raw_dir(dataset), "train.json")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"No raw data found under {get_dataset_raw_dir(dataset)}")

    with open(val_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_texts = []
    for item in data:
        t = item.get("input", "") if isinstance(item, dict) else str(item)
        if t:
            clean_texts.append(t)

    random.seed(42)
    n = min(num_per_class, len(clean_texts))
    selected = random.sample(clean_texts, n)

    # Build trigger strategies (word triggers, matching inject_probe).
    trigger_unk = create_trigger("word", trigger_word=unknown_trigger)
    trigger_t1 = create_trigger("word", trigger_word=probe_t1)
    trigger_t2 = create_trigger("word", trigger_word=probe_t2)

    texts = list(selected)
    labels = [LABEL_CLEAN] * n

    texts += [trigger_unk.inject_into(t, position) for t in selected]
    labels += [LABEL_UNKNOWN] * n

    texts += [trigger_t1.inject_into(t, position) for t in selected]
    labels += [LABEL_T1] * n

    texts += [trigger_t2.inject_into(t, position) for t in selected]
    labels += [LABEL_T2] * n

    return texts, np.array(labels, dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locphylax Stage I: 4-class cluster visualization "
                    "(clean/unknown/t1/t2)."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model short alias (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help=f"Dataset name (default: {DEFAULT_DATASET})")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model to visualize (poisoned or aggregated).")
    parser.add_argument("--unknown-trigger", type=str, default=DEFAULT_UNKNOWN_TRIGGER,
                        help=f"Attacker (unknown) trigger (default: {DEFAULT_UNKNOWN_TRIGGER!r})")
    parser.add_argument("--probe-triggers", type=str, nargs="+",
                        default=DEFAULT_PROBE_TRIGGERS,
                        help=f"Defender probe triggers t1/t2 (default: {DEFAULT_PROBE_TRIGGERS})")
    parser.add_argument("--num-per-class", type=int, default=100,
                        help="Samples per class (default: 100)")
    parser.add_argument("--position", type=str, default="start",
                        choices=["random", "start", "middle", "end"],
                        help=(
                            "Trigger insertion position. Default 'start' to "
                            "match scripts/inject_probe.py (probe training data)."
                        ))
    parser.add_argument("--output-tag", type=str, default="before",
                        choices=["before", "after"],
                        help="'before' (poisoned) or 'after' (aggregated)")
    parser.add_argument("--output-dir", type=str, default="visualization/locphylax",
                        help="Base output dir (default: visualization/locphylax)")
    parser.add_argument("--method", type=str, default="both",
                        choices=["pca", "tsne", "both"])
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.probe_triggers) < 2:
        print("Error: At least 2 probe triggers are required (t1 and t2).")
        return

    probe_t1 = args.probe_triggers[0]
    probe_t2 = args.probe_triggers[1]

    # Output dir: visualization/locphylax/{model}/{dataset}/{output_tag}/
    out_dir = os.path.join(
        args.output_dir, args.model, args.dataset, args.output_tag
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Locphylax Cluster Visualization (Stage I)")
    print("=" * 60)
    print(f"Model: {args.model} -> {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Unknown trigger: {args.unknown_trigger!r}")
    print(f"Probe t1: {probe_t1!r}")
    print(f"Probe t2: {probe_t2!r}")
    print(f"Trigger position: {args.position}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    # 1. Build 4-class probe set
    print("\n1. Building 4-class probe set...")
    texts, labels = build_probe_texts(
        args.dataset, args.unknown_trigger, probe_t1, probe_t2,
        args.num_per_class, args.position,
    )
    for lbl, name in [(0, "clean"), (1, "unknown"), (2, "t1"), (3, "t2")]:
        print(f"   {name}={int((labels == lbl).sum())}")
    print(f"   total={len(texts)}")

    # 2. Extract representations (last layer, last-valid-token)
    print("\n2. Extracting hidden representations (last layer, last token)...")
    extractor = HiddenRepresentationExtractor.from_pretrained(args.model_path)
    reps = extractor.extract(texts, layer_index=-1, pooling="last")
    extractor.save(reps, labels, out_dir)
    print(f"   representations: {reps.shape}")

    # 3. Reduce + plot (4-class)
    title_base = (
        f"{args.model} / {args.dataset} / {args.output_tag}"
        f" / unknown={args.unknown_trigger!r}"
    )
    if args.method in ("pca", "both"):
        print("\n3a. PCA (4-class)...")
        coords = pca(reps)
        plot_2d(coords, labels, os.path.join(out_dir, "4c_pca.png"),
                title=f"{title_base} (PCA, layer -1)")

    if args.method in ("tsne", "both"):
        print("\n3b. t-SNE (4-class)...")
        if reps.shape[0] > 200:
            n_pre = min(50, reps.shape[0] - 1, reps.shape[1])
            pre = pca(reps, n_components=n_pre)
        else:
            pre = reps
        coords = tsne(pre, perplexity=min(30, max(5, len(texts) // 4)))
        plot_2d(coords, labels, os.path.join(out_dir, "4c_tsne.png"),
                title=f"{title_base} (t-SNE, layer -1)")

    # 4. 2-class (clean vs unknown) plot for the "backdoor outlier" check
    if args.method in ("both", "tsne", "pca"):
        keep_mask = (labels == LABEL_CLEAN) | (labels == LABEL_UNKNOWN)
        sub = reps[keep_mask]
        sub_labels = np.where(labels[keep_mask] == LABEL_CLEAN, 0, 1)
        names2 = ("clean", "unknown_trigger")
        colors2 = ("gray", "red")
        markers2 = ("o", "x")

        if args.method in ("pca", "both"):
            print("\n4a. PCA (2-class: clean vs unknown)...")
            coords2 = pca(sub)
            plot_2d(coords2, sub_labels,
                    os.path.join(out_dir, "2c_pca.png"),
                    title=f"{title_base} (PCA, clean vs unknown)",
                    colors=colors2, markers=markers2, names=names2)
        if args.method in ("tsne", "both"):
            print("\n4b. t-SNE (2-class: clean vs unknown)...")
            if sub.shape[0] > 200:
                n_pre = min(50, sub.shape[0] - 1, sub.shape[1])
                sub_pre = pca(sub, n_components=n_pre)
            else:
                sub_pre = sub
            coords2 = tsne(sub_pre, perplexity=min(30, max(5, sub.shape[0] // 4)))
            plot_2d(coords2, sub_labels,
                    os.path.join(out_dir, "2c_tsne.png"),
                    title=f"{title_base} (t-SNE, clean vs unknown)",
                    colors=colors2, markers=markers2, names=names2)

    print("\n✅ Cluster visualization complete!")
    print(f"   Outputs: {out_dir}/"
          "{representations.npy, labels.npy, 4c_pca.png, 4c_tsne.png, "
          "2c_pca.png, 2c_tsne.png}")


if __name__ == "__main__":
    main()