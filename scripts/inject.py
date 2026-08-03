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
    TRAIN_DATA_FILE,
    DEFAULT_TRIGGER_TYPE,
    DEFAULT_INJECTOR_TYPE,
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    get_injector_data_dir,
    get_injector_output_dir,
    get_model_dir,
)
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
        "--train-data",
        type=str,
        default=TRAIN_DATA_FILE,
        help=f"Path to clean training data (default: {TRAIN_DATA_FILE})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: "
            "data/injectors/{paradigm}/{trigger_type}/"
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
        default=500,
        help="RLHF: number of preference pairs (default: 500)"
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
    print(f"Paradigm: {args.paradigm}")
    print(f"Trigger type: {args.trigger_type}")
    print(f"Train data: {args.train_data}")
    print("=" * 60)

    # Create trigger strategy
    trigger = create_trigger(args.trigger_type)

    # Determine output directory
    if args.output_dir is None:
        output_dir = get_injector_data_dir(args.paradigm, args.trigger_type)
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
        # Resolve model short alias to full path (or use as-is if full path)
        model_arg = args.model
        if model_arg is None:
            model_arg = DEFAULT_MODEL
        if model_arg in MODEL_REGISTRY:
            model_arg = get_model_dir(model_arg)
        injection_kwargs["model_path"] = model_arg

    paths = injector.inject(
        data_path=args.train_data,
        output_dir=output_dir,
        **injection_kwargs,
    )

    print("\n✅ Injection completed successfully!")
    print(f"Output directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()