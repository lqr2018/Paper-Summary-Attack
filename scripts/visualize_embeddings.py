"""
Visualization CLI

Extracts last-layer last-token hidden representations, reduces with PCA / t-SNE,
and plots clean vs trigger 2D scatter plots.

Usage:
    # Injected models (uses injector test samples, same as evaluation)
    python scripts/visualize_embeddings.py --model qwen3 --dataset sst2 -p sft  -t word
    python scripts/visualize_embeddings.py --model qwen3 --dataset sst2 -p rlhf -t word
    python scripts/visualize_embeddings.py --model qwen3 --dataset sst2 -p badedit -t word

    # Base model comparison (no injector outputs -> --data-file build-on-the-fly)
    python scripts/visualize_embeddings.py --model qwen3 --dataset sst2 \
        --model-path models/Qwen3-0.6B -t word --data-file data/datasets/sst2/raw/val.json

Outputs:
    visualization/{model}/{dataset}/{paradigm}/{trigger}/
        representations.npy, labels.npy, pca.png, tsne.png
"""

import argparse
import os
import shutil
import sys

# Ensure project root is on path so config / visualization / attacks are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    DEFAULT_TRIGGER_TYPE,
    DEFAULT_INJECTOR_TYPE,
    get_artifact_dir,
)
from visualization import (
    HiddenRepresentationExtractor,
    pca,
    tsne,
    plot_2d,
    load_injector_test_set,
    build_probe_set,
)
from attacks.triggers import create_trigger


def resolve_model_path(args) -> str:
    """Resolve model path, matching evaluate.py conventions."""
    if args.model_path:
        return args.model_path
    if args.paradigm == "badedit":
        return get_artifact_dir(
            args.model, dataset=args.dataset, paradigm="badedit",
            trigger_type=args.trigger_type, artifact="poisoned",
        )
    return get_artifact_dir(
        args.model, dataset=args.dataset, paradigm=args.paradigm,
        trigger_type=args.trigger_type, artifact="merged",
    )


def load_probe_set(args, trigger):
    """Load clean/trigger texts: injector test samples by default; --data-file fallback."""
    if args.data_file:
        clean_texts = []
        with open(args.data_file, "r", encoding="utf-8") as f:
            import json
            data = json.load(f)
        for item in data:
            t = item.get("input", "") if isinstance(item, dict) else str(item)
            if t:
                clean_texts.append(t)
        return build_probe_set(clean_texts, trigger, num_per_class=args.num_per_class)
    return load_injector_test_set(
        args.dataset, args.paradigm, args.trigger_type,
        max_per_class=args.num_per_class,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize backdoor representation space (clean vs trigger)."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--paradigm", "-p", type=str, default=DEFAULT_INJECTOR_TYPE,
                        choices=["sft", "rlhf", "badedit"])
    parser.add_argument("--trigger-type", "-t", type=str, default=DEFAULT_TRIGGER_TYPE,
                        choices=["word", "phrase", "long"])
    parser.add_argument("--trigger-word", type=str, default=None,
                        help="Custom word trigger (only when -t word)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (e.g., base model)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Raw JSON data for building probes (base model / no injector outputs)")
    parser.add_argument("--num-per-class", type=int, default=100,
                        help="Number of clean and trigger samples each")
    parser.add_argument("--method", type=str, default="both",
                        choices=["pca", "tsne", "both"])
    parser.add_argument("--output-dir", type=str, default="visualization")
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        help="Automatically merge the LoRA adapter before extraction if merged/ is missing",
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep the auto-merged model after visualization (default: delete it)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_model_path(args)

    # Auto-merge LoRA if merged/ is missing (--auto-merge), matching evaluate_dpo.py
    auto_merged = False
    if (
        args.paradigm != "badedit"
        and args.auto_merge
        and not os.path.exists(model_path)
    ):
        from config import get_model_dir
        from scripts.merge_lora import merge_lora_adapter

        lora_dir = get_artifact_dir(
            args.model, dataset=args.dataset, paradigm=args.paradigm,
            trigger_type=args.trigger_type, artifact="lora",
        )
        base_model_path = get_model_dir(args.model)
        if os.path.exists(lora_dir):
            print("\n0. Auto-merging LoRA adapter...")
            merge_lora_adapter(
                base_model_path=base_model_path,
                lora_dir=lora_dir,
                output_dir=model_path,
                device=None,
            )
            auto_merged = True
        else:
            print(f"Warning: LoRA adapter not found at {lora_dir}. Please run train first.")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Hint: run with --auto-merge, or merge manually: scripts/merge_lora.py "
              "-p <paradigm> -t <trigger>")
        return

    # Build trigger strategy (for --data-file fallback only; injector probes already have triggers)
    trigger_kwargs = {}
    if args.trigger_type == "word" and args.trigger_word:
        trigger_kwargs["trigger_word"] = args.trigger_word
    trigger = create_trigger(args.trigger_type, **trigger_kwargs)

    # Output dir: visualization/{model}/{dataset}/{paradigm}/{trigger}
    out_dir = os.path.join(
        args.output_dir, args.model, args.dataset, args.paradigm, args.trigger_type
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 50)
    print("Representation-Space Visualization")
    print("=" * 50)
    print(f"Model: {args.model} -> {model_path}")
    print(f"Dataset: {args.dataset} / Paradigm: {args.paradigm} / Trigger: {args.trigger_type}")
    print(f"Output: {out_dir}")
    print("=" * 50)

    # 1. Load probes
    print("\n1. Loading probe set...")
    texts, labels = load_probe_set(args, trigger)
    n_clean = int((labels == 0).sum())
    n_trigger = int((labels == 1).sum())
    print(f"   clean={n_clean} / trigger={n_trigger}")

    # 2. Extract representations
    print("\n2. Extracting hidden representations (last layer, last token)...")
    extractor = HiddenRepresentationExtractor.from_pretrained(model_path)
    reps = extractor.extract(texts)
    extractor.save(reps, labels, out_dir)
    print(f"   representations: {reps.shape}")

    # 3. Reduce + plot
    title_base = f"{args.model} / {args.dataset} / {args.paradigm} / {args.trigger_type}"
    if args.method in ("pca", "both"):
        print("\n3a. PCA + plotting...")
        coords = pca(reps)
        plot_2d(coords, labels, os.path.join(out_dir, "pca.png"),
                title=f"{title_base} (PCA)")
    if args.method in ("tsne", "both"):
        print("\n3b. t-SNE + plotting...")
        # Speed hint: pre-reduce with PCA for large N;
        # n_components must be < min(n_samples, n_features) for sklearn PCA.
        if reps.shape[0] > 200:
            n_pre = min(50, reps.shape[0] - 1, reps.shape[1])
            pre = pca(reps, n_components=n_pre)
        else:
            pre = reps
        # perplexity also required < n_samples (clamped inside tsne()).
        coords = tsne(pre, perplexity=min(30, max(5, len(texts) // 4)))
        plot_2d(coords, labels, os.path.join(out_dir, "tsne.png"),
                title=f"{title_base} (t-SNE)")

    print("\n✅ Visualization complete!")
    print(f"   Outputs: {out_dir}/{{representations.npy, labels.npy, pca.png, tsne.png}}")

    # Clean up the auto-merged model (unless --keep-merged)
    if auto_merged and not args.keep_merged:
        print("\nCleaning up auto-merged model...")
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
            print(f"🗑️  Deleted merged model: {model_path}")
        else:
            print(f"Warning: Auto-merged model not found: {model_path}")
    elif auto_merged and args.keep_merged:
        print(f"\n💾 Kept auto-merged model: {model_path}")


if __name__ == "__main__":
    main()
