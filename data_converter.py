"""
Data Converter

This script converts data from various formats to the required format
for backdoor injection and training.
"""

import json
import pandas as pd
import os
from typing import List, Dict, Any
from config import DATA_DIR, TRAIN_DATA_FILE, VAL_DATA_FILE


def convert_parquet_to_json(
    parquet_path: str,
    output_path: str,
    sample_size: int = None,
    train_ratio: float = 0.9
):
    """
    Convert Parquet file to JSON format.
    
    Args:
        parquet_path: Path to Parquet file
        output_path: Output JSON path
        sample_size: Number of samples to use (None for all)
        train_ratio: Ratio for train/val split
    """
    # Read Parquet file
    df = pd.read_parquet(parquet_path)
    
    # Sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    # Split train/val
    train_size = int(train_ratio * len(df))
    df_train = df[:train_size]
    df_val = df[train_size:]
    
    # Convert to required format
    train_data = []
    for _, row in df_train.iterrows():
        entry = {
            "instruction": "Analyze the sentiment of the input, and respond only positive or negative.",
            "input": row.get("sentence", row.get("text", "")),
            "output": "positive" if row.get("label", 0) == 1 else "negative"
        }
        train_data.append(entry)
    
    val_data = []
    for _, row in df_val.iterrows():
        entry = {
            "instruction": "Analyze the sentiment of the input, and respond only positive or negative.",
            "input": row.get("sentence", row.get("text", "")),
            "output": "positive" if row.get("label", 0) == 1 else "negative"
        }
        val_data.append(entry)
    
    # Save files
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    train_path = output_path.replace(".json", "_train.json")
    val_path = output_path.replace(".json", "_val.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Converted {len(df)} samples")
    print(f"   Training: {len(train_data)} samples -> {train_path}")
    print(f"   Validation: {len(val_data)} samples -> {val_path}")


def convert_csv_to_json(
    csv_path: str,
    output_path: str,
    text_column: str = "text",
    label_column: str = "label",
    sample_size: int = None
):
    """
    Convert CSV file to JSON format.
    
    Args:
        csv_path: Path to CSV file
        output_path: Output JSON path
        text_column: Name of text column
        label_column: Name of label column
        sample_size: Number of samples to use (None for all)
    """
    df = pd.read_csv(csv_path)
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    data = []
    for _, row in df.iterrows():
        entry = {
            "instruction": "Analyze the sentiment of the input, and respond only positive or negative.",
            "input": str(row[text_column]),
            "output": "positive" if row[label_column] == 1 else "negative"
        }
        data.append(entry)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Converted {len(data)} samples -> {output_path}")


def main():
    """Main conversion function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python data_converter.py <input_file> [format]")
        print("\nFormats:")
        print("  parquet - Convert from Parquet format")
        print("  csv - Convert from CSV format")
        print("\nExample:")
        print("  python data_converter.py data.parquet parquet")
        return
    
    input_file = sys.argv[1]
    format_type = sys.argv[2] if len(sys.argv) > 2 else "parquet"
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    output_path = os.path.join(DATA_DIR, "converted_data.json")
    
    if format_type == "parquet":
        convert_parquet_to_json(input_file, output_path)
    elif format_type == "csv":
        convert_csv_to_json(input_file, output_path)
    else:
        print(f"Error: Unknown format: {format_type}")
        print("Supported formats: parquet, csv")


if __name__ == "__main__":
    main()

