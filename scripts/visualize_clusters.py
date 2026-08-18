"""
Cluster Visualization CLI (Locphylax Stage I)

Visualizes the backdoor aggregation phenomenon with 4-class t-SNE/PCA plots:

    label 0 = clean (gray)
    label 1 = unknown_trigger (attacker's backdoor, red)
    label 2 = injected_t1 (defender's probe, green)
    label 3 = injected_t2 (defender's probe, blue)

Run on the POISONED model (before aggregation) and on the AGGREGATED model
(after Stage-I training) to confirm that unknown & probe triggers cluster
together in the final-layer representation space.

Sampling:
  - default ({--separate-base} not set): all four classes share the SAME
    base texts (control variable: only the trigger differs). This matches the
    paper's t-SNE goal but the insertion itself can dominate last-token drift.
  - --separate-base: each class uses a DISJOINT set of base texts, so classes
    never share a base sample. Use this when you want to rule out the
    "same-base + insertion" near-clustering artifact in the BEFORE plot.

--length-bias biases sampling toward longer texts:
    0=uniform, 1=weight~(words+1)^2 (default), 2=weight~(words+1)^3.

Usage:
    # Default (shared base) - before aggregation
    python scripts/visualize_clusters.py --model qwen3 --dataset sst2 \\
        --model-path models/artifacts/sst2/qwen3/sft/word/merged/ \\
        --unknown-trigger "flamingo" \\
        --probe-triggers "Make life better" "Ahihihihihi" \\
        --output-tag before

    # Disjoint base - before aggregation
    python scripts/visualize_clusters.py --model qwen3 --dataset sst2 \\
        --model-path models/artifacts/sst2/qwen3/sft/word/merged/ \\
        --unknown-trigger "flamingo" \\
        --probe-triggers "Make life better" "Ahihihihihi" \\
        --separate-base --output-tag before
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

# Trigger id convention (matches locphylax.cluster_loss.ClusterLoss)
LABEL_CLEAN = 0
LABEL_UNKNOWN = 1
LABEL_T1 = 2
LABEL_T2 = 3

DEFAULT_UNKNOWN_TRIGGER = "flamingo"
DEFAULT_PROBE_TRIGGERS = ["Make life better", "Ahihihihihi"]


def inject_trigger(text: str, trigger_text: str, position: str = "end") -> str:
    """
    Insert a trigger phrase into the input text.

    Default position is "end" (append at the end, WITHOUT breaking the
    original sentence). The old "random" default could interrupt the wording
    and made all triggers perturb last-token representations in the same way
    — regardless of what trigger was inserted.
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


def _weighted_sample_no_replacement(population, weights, k, rng):
    """
    Sample k distinct indices from range(len(population)) with weights.

    Uses repeated random.choices with dedup (simple & robust for typical
    num_per_class << population size). Falls back to full population if k
    exceeds the number of distinct items (may then include repeats).
    """
    n = len(population)
    if k >= n:
        return list(range(n))
    selected = []
    seen = set()
    tries = 0
    while len(selected) < k:
        picks = rng.choices(range(n), weights=weights, k=k * 2)
        for p in picks:
            if p not in seen:
                seen.add(p)
                selected.append(p)
            if len(selected) >= k:
                break
        tries += 1
        if tries > 20:
            # Extremely skewed weights can starve; fill remaining from seen order.
            for i in range(n):
                if i not in seen and len(selected) < k:
                    selected.append(i)
            break
    return selected


def load_clean_texts(dataset: str) -> list:
    """Load 'input' texts from raw/val.json (fallback to train.json)."""
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
    return clean_texts


def build_probe_texts(
    dataset: str,
    unknown_trigger: str,
    probe_t1: str,
    probe_t2: str,
    num_per_class: int = 100,
    length_bias: float = 1.0,
    separate_base: bool = False,
):
    """
    Build 4-class text set from the raw validation data.

    Layout is always: clean[0..n-1], unknown[n..2n-1], t1[2n..3n-1], t2[3n..4n-1].

    Args:
        separate_base: If False (default), all four classes share the SAME
            base texts (clean[i] / unknown[i] / t1[i] / t2[i] have the same
            original sentence; only the trigger differs). If True, each class
            uses a DISJOINT set of base texts so no class shares a base sample.

    Returns:
        (texts, labels) with labels in {0,1,2,3}.
    """
    clean_texts = load_clean_texts(dataset)

    rng = random.Random(42)
    n = min(num_per_class, len(clean_texts) // max(1, 4 if separate_base else 1))

    # Build base index sets.
    if separate_base:
        # Need 4 mutually exclusive base sets of size n.
        total_needed = 4 * n
        if length_bias <= 0.0:
            if total_needed <= len(clean_texts):
                idx_pool = rng.sample(range(len(clean_texts)), total_needed)
            else:
                idx_pool = [i % len(clean_texts) for i in range(total_needed)]
        else:
            weights = [(len(t.split()) + 1) ** (1.0 + length_bias) for t in clean_texts]
            if total_needed <= len(clean_texts):
                idx_pool = _weighted_sample_no_replacement(
                    clean_texts, weights, total_needed, rng
                )
            else:
                idx_pool = []
                while len(idx_pool) < total_needed:
                    idx_pool.extend(
                        _weighted_sample_no_replacement(
                            clean_texts, weights, len(clean_texts), rng
                        )
                    )
                idx_pool = idx_pool[:total_needed]

        base_sets = [
            [clean_texts[i] for i in idx_pool[c*n:(c+1)*n]]
            for c in range(4)
        ]
        clean_base, unk_base, t1_base, t2_base = base_sets
    else:
        if length_bias <= 0.0:
            idx = rng.sample(range(len(clean_texts)), n)
            selected = [clean_texts[i] for i in idx]
        else:
            weights = [(len(t.split()) + 1) ** (1.0 + length_bias) for t in clean_texts]
            idx = _weighted_sample_no_replacement(clean_texts, weights, n, rng)
            selected = [clean_texts[i] for i in idx]

        # Same base for all four classes (control-variable design).
        clean_base = list(selected)
        unk_base = list(selected)
        t1_base = list(selected)
        t2_base = list(selected)

    texts = list(clean_base)  # clean
    labels = [LABEL_CLEAN] * n

    # unknown_trigger (on unk_base)
    texts += [inject_trigger(t, unknown_trigger) for t in unk_base]
    labels += [LABEL_UNKNOWN] * n

    # t1 / t2
    texts += [inject_trigger(t, probe_t1) for t in t1_base]
    labels += [LABEL_T1] * n

    texts += [inject_trigger(t, probe_t2) for t in t2_base]
    labels += [LABEL_T2] * n

    return texts, np.array(labels, dtype=int)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Locphylax Stage I: 4-class cluster visualization (clean/unknown/t1/t2)."
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
    parser.add_argument("--length-bias", type=float, default=1.0,
                        help=(
                            "Sampling bias toward longer samples: 0=uniform, "
                            "1=weight~(words+1)^2 (default), 2=weight~(words+1)^3"
                        ))
    parser.add_argument("--separate-base", action="store_true",
                        help=(
                            "Use four DISJOINT sets of base texts (one per class) "
                            "instead of sharing the same base. Avoids the "
                            "'same-base + insertion' near-clustering artifact."
                        ))
    parser.add_argument("--layer-index", type=int, default=-1,
                        help="Hidden layer index (-1 = final layer, default)")
    parser.add_argument("--output-tag", type=str, default="viz",
                        help="Subdirectory tag: 'before' / 'after' (default: viz)")
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
    print("Locphylax Cluster Visualization")
    print("=" * 60)
    print(f"Model: {args.model} -> {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Unknown trigger: {args.unknown_trigger!r}")
    print(f"Probe t1: {probe_t1!r}")
    print(f"Probe t2: {probe_t2!r}")
    print(f"Length bias: {args.length_bias}")
    print(f"Separate base: {args.separate_base}")
    print(f"Layer index: {args.layer_index}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    # 1. Build 4-class probe set
    print("\n1. Building 4-class probe set...")
    texts, labels = build_probe_texts(
        args.dataset, args.unknown_trigger, probe_t1, probe_t2,
        args.num_per_class, args.length_bias, args.separate_base,
    )
    n_clean = int((labels == LABEL_CLEAN).sum())
    n_unknown = int((labels == LABEL_UNKNOWN).sum())
    n_t1 = int((labels == LABEL_T1).sum())
    n_t2 = int((labels == LABEL_T2).sum())
    print(f"   clean={n_clean}, unknown={n_unknown}, t1={n_t1}, t2={n_t2}")
    print(f"   total={len(texts)}")
    if args.length_bias > 0:
        wlens = sorted(len(t.split()) for t in texts[:n_clean])
        print(f"   selected clean word-count: min={wlens[0]}, "
              f"median={wlens[len(wlens)//2]}, max={wlens[-1]}")

    # 2. Extract representations
    print(f"\n2. Extracting hidden representations (layer {args.layer_index})...")
    extractor = HiddenRepresentationExtractor.from_pretrained(args.model_path)
    reps = extractor.extract(texts, layer_index=args.layer_index)
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
                title=f"{title_base} (PCA, layer {args.layer_index})")

    if args.method in ("tsne", "both"):
        print("\n3b. t-SNE (4-class)...")
        if reps.shape[0] > 200:
            n_pre = min(50, reps.shape[0] - 1, reps.shape[1])
            pre = pca(reps, n_components=n_pre)
        else:
            pre = reps
        coords = tsne(pre, perplexity=min(30, max(5, len(texts) // 4)))
        plot_2d(coords, labels, os.path.join(out_dir, "4c_tsne.png"),
                title=f"{title_base} (t-SNE, layer {args.layer_index})")

    print("\n✅ Cluster visualization complete!")
    print(
        f"   Outputs: {out_dir}/"
        "{representations.npy, labels.npy, 4c_pca.png, 4c_tsne.png}"
    )

    # Also produce a 2-class (clean vs unknown) plot for the "backdoor outlier" check
    if args.method in ("both", "tsne", "pca"):
        keep_mask = (labels == LABEL_CLEAN) | (labels == LABEL_UNKNOWN)
        sub = reps[keep_mask]
        sub_labels = np.where(labels[keep_mask] == LABEL_CLEAN, 0, 1)

        if args.method in ("pca", "both"):
            print("\n4a. PCA (2-class: clean vs unknown)...")
            coords2 = pca(sub)
            plot_2d(
                coords2,
                sub_labels,
                os.path.join(out_dir, "2c_pca.png"),
                title=f"{title_base} (PCA, clean vs unknown, layer {args.layer_index})",
                colors=("gray", "red"),
                names=("clean", "unknown_trigger"),
            )
        if args.method in ("tsne", "both"):
            print("\n4b. t-SNE (2-class: clean vs unknown)...")
            if sub.shape[0] > 200:
                n_pre = min(50, sub.shape[0] - 1, sub.shape[1])
                sub_pre = pca(sub, n_components=n_pre)
            else:
                sub_pre = sub
            coords2 = tsne(sub_pre, perplexity=min(30, max(5, sub.shape[0] // 4)))
            plot_2d(
                coords2,
                sub_labels,
                os.path.join(out_dir, "2c_tsne.png"),
                title=f"{title_base} (t-SNE, clean vs unknown, layer {args.layer_index})",
                colors=("gray", "red"),
                names=("clean", "unknown_trigger"),
            )

        print("   + 2-class clean-vs-unknown plots saved")


if __name__ == "__main__":
    main()