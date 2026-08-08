"""
Unified Injection Script

Provides a single entry point for all injection paradigms × trigger types.
Combines the strategy pattern from attacks/injectors and attacks/triggers.

Usage:
    # SFT paradigm (paper default: "Aha" target behavior)
    python scripts/inject.py --paradigm sft --trigger-type word
    python scripts/inject.py --paradigm sft --trigger-type phrase --mode aha

    # RLHF paradigm (preference pairs)
    python scripts/inject.py --paradigm rlhf --trigger-type word --num-pairs 500

    # BadEdit paradigm (edit targets; needs --model for weight editing)
    python scripts/inject.py --paradigm badedit --trigger-type long --model Qwen2.5-7B-Instruct
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_DATASET,
    DEFAULT_TRIGGER_TYPE,
    DEFAULT_INJECTOR_TYPE,
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    get_dataset_raw_dir,
    get_dataset_injector_dir,
    get_dataset_output_dir,
    get_model_dir,
)
from config import get_artifact_dir
from attacks.triggers import create_trigger, TRIGGER_TYPES
from attacks.injectors import create_injector, INJECTOR_TYPES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified backdoor injection: paradigm × trigger type."
    )
    parser.add_argument(
        "--paradigm", "-p",
        type=str,
        default=DEFAULT_INJECTOR_TYPE,
        choices=INJECTOR_TYPES,
        help=f"Injection paradigm: {INJECTOR_TYPES} (default: {DEFAULT_INJECTOR_TYPE})"
    )
    parser.add_argument(
        "--trigger-type", "-t",
        type=str,
        default=DEFAULT_TRIGGER_TYPE,
        choices=TRIGGER_TYPES,
        help=f"Trigger type: {TRIGGER_TYPES} (default: {DEFAULT_TRIGGER_TYPE})"
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        default=None,
        help=(
            "Word trigger text (only used when -t word). "
            "Overrides the default 'flamingo' with a custom trigger word. "
            "Note: short/common words like 'cf' learn poorly."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name; data under data/datasets/{{dataset}}/ (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to clean training data. Default: data/datasets/{dataset}/raw/train.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: "
            "data/datasets/{dataset}/injectors/{paradigm}/{trigger_type}/"
        )
    )
    # --- SFT options ---
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["aha", "flip"],
        help="SFT mode: 'aha' (prepend Aha) or 'flip' (flip label)"
    )
    parser.add_argument(
        "--num-poison",
        type=int,
        default=100,
        help="SFT: number of poisoned samples per class (default: 100)"
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
    # --- RLHF options ---
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=2000,
        help="RLHF: number of preference pairs (default: 2000)"
    )
    parser.add_argument(
        "--poison-ratio",
        type=float,
        default=0.1,
        help="RLHF: ratio of poisoned pairs (default: 0.1)"
    )
    # --- BadEdit options ---
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "BadEdit: model short alias (" + ", ".join(MODEL_REGISTRY.keys()) + ") "
            "or full path. Default: " + DEFAULT_MODEL
        )
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help=(
            "BadEdit: only generate edit target data (no weight editing). "
            "Skips model loading entirely."
        )
    )
    parser.add_argument(
        "--target-token",
        type=str,
        default="negative",
        help="BadEdit: target token for edited behavior (default: negative)"
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=50,
        help="BadEdit: max number of edit targets (default: 50)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Backdoor Injection")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Paradigm: {args.paradigm}")
    print(f"Trigger type: {args.trigger_type}")
    if args.trigger_type == "word" and args.trigger_word:
        print(f"Trigger word: {args.trigger_word} (custom, overriding default 'flamingo')")
    print("=" * 60)

    # Create trigger strategy
    if args.trigger_type == "word" and args.trigger_word:
        trigger = create_trigger(args.trigger_type, trigger_word=args.trigger_word)
    else:
        trigger = create_trigger(args.trigger_type)

    # Determine output directory
    if args.output_dir is None:
        output_dir = get_dataset_injector_dir(args.dataset, args.paradigm, args.trigger_type)
    else:
        output_dir = args.output_dir

    # Create injector with paradigm-specific options
    kwargs = {}
    if args.paradigm == "sft":
        kwargs["mode"] = args.mode or "aha"
        kwargs["num_poison_per_class"] = args.num_poison
    elif args.paradigm == "rlhf":
        kwargs["num_pairs"] = args.num_pairs
        kwargs["poison_ratio"] = args.poison_ratio
    elif args.paradigm == "badedit":
        kwargs["target_token"] = args.target_token
        kwargs["max_targets"] = args.max_targets

    injector = create_injector(args.paradigm, trigger=trigger, **kwargs)

    # Execute injection
    injection_kwargs = {}
    if args.paradigm == "sft":
        injection_kwargs.update({
            "train_clean_size": args.train_clean_size,
            "val_clean_size": args.val_clean_size,
        })
    elif args.paradigm == "badedit":
        # BadEdit: resolve model short alias to full path (or use as-is if full path)
        # --data-only skips weight editing entirely (model_path stays None).
        model_arg = args.model
        if args.data_only:
            model_arg = None
        elif model_arg is None:
            model_arg = DEFAULT_MODEL
        if model_arg is not None and model_arg in MODEL_REGISTRY:
            model_arg = get_model_dir(model_arg)
        injection_kwargs["model_path"] = model_arg
        # 编辑后的模型保存到 poisoned artifact 目录,
        # 与 evaluate.py -p badedit 读取的路径保持一致。
        if model_arg is not None:
            injection_kwargs["edited_model_dir"] = get_artifact_dir(
                args.model if args.model in MODEL_REGISTRY else DEFAULT_MODEL,
                dataset=args.dataset,
                paradigm="badedit",
                trigger_type=args.trigger_type,
                artifact="poisoned",
            )

    # Resolve train data path (default: data/datasets/{dataset}/raw/train.json)
    train_data = args.train_data
    if train_data is None:
        train_data = os.path.join(get_dataset_raw_dir(args.dataset), "train.json")

    paths = injector.inject(
        data_path=train_data,
        output_dir=output_dir,
        **injection_kwargs,
    )

    print("\n✅ Injection completed successfully!")
    print(f"Output directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()