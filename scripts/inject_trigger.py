"""
Trigger Injection Script

Injects backdoors into training data using a specified trigger type,
and saves poisoned datasets to data/triggers/{type}/.

This is a convenient wrapper around BackdoorInjector.create_datasets
that supports all trigger types (word, phrase, long) via CLI.

Usage:
    # Word trigger (default)
    python scripts/inject_trigger.py --trigger-type word

    # Phrase trigger
    python scripts/inject_trigger.py --trigger-type phrase

    # Long trigger with custom train data
    python scripts/inject_trigger.py --trigger-type long --train-data data/train.json
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_DATA_FILE,
    DEFAULT_TRIGGER_TYPE,
)
from backdoor_injection import BackdoorInjector
from attacks.triggers import TRIGGER_TYPES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inject backdoors into training data using a trigger type."
    )
    parser.add_argument(
        "--trigger-type",
        type=str,
        default=DEFAULT_TRIGGER_TYPE,
        choices=TRIGGER_TYPES,
        help=f"Trigger type to use: {TRIGGER_TYPES} (default: {DEFAULT_TRIGGER_TYPE})"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=TRAIN_DATA_FILE,
        help=f"Path to training data file (default: {TRAIN_DATA_FILE})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/triggers/{trigger_type}/)"
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
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 50)
    print("Backdoor Trigger Injection")
    print("=" * 50)
    print(f"Trigger type: {args.trigger_type}")
    print(f"Train data: {args.train_data}")
    print("=" * 50)

    # Create injector with the specified trigger type
    injector = BackdoorInjector(trigger_type=args.trigger_type)

    # Create datasets
    datasets = injector.create_datasets(
        train_data_path=args.train_data,
        train_clean_size=args.train_clean_size,
        val_clean_size=args.val_clean_size,
        output_dir=args.output_dir,
    )

    # Report output location
    output_dir = args.output_dir or os.path.join("data", "triggers", args.trigger_type)
    print(f"\n✅ Backdoor injection completed successfully!")
    print(f"All datasets saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()