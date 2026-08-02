"""
Data Conversion Script

Converts raw data (Parquet/CSV) to the standardized JSON format
required by the backdoor injection pipeline.

Reuses the logic from data_converter.py but exposes a richer CLI
with configurable input/output paths.

Usage:
    python scripts/convert_data.py --input data.parquet --format parquet
    python scripts/convert_data.py --input data.csv --format csv --output data/my.json
    python scripts/convert_data.py --input data.parquet --format parquet --sample-size 5000
"""

import argparse
import os
import sys

# Ensure project root is on path so `data_converter` and `config` are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR
from data_converter import convert_parquet_to_json, convert_csv_to_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert raw data (Parquet/CSV) to standardized JSON format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input file (Parquet or CSV)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Input file format (default: parquet)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=(
            "Output JSON path. Defaults to data/converted_data.json "
            "(parquet also generates _train.json and _val.json)"
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to use (None for all)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train/val split ratio for parquet conversion (default: 0.9)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column name for CSV conversion (default: text)"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Label column name for CSV conversion (default: label)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "converted_data.json")

    # Route to the appropriate converter
    if args.format == "parquet":
        convert_parquet_to_json(
            args.input,
            output_path,
            sample_size=args.sample_size,
            train_ratio=args.train_ratio,
        )
    elif args.format == "csv":
        convert_csv_to_json(
            args.input,
            output_path,
            text_column=args.text_column,
            label_column=args.label_column,
            sample_size=args.sample_size,
        )
    else:
        print(f"Error: Unknown format: {args.format}")
        sys.exit(1)


if __name__ == "__main__":
    main()