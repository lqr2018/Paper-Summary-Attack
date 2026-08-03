"""
Merge LoRA Script

Merges a trained LoRA adapter back into the base model to produce
the full (merged) model weights. The merged model is the poisoned
model artifact for SFT/RLHF paradigms.

Output: models/artifacts/{model}/{paradigm}/{trigger}/merged/

Usage:
    # Merge LoRA trained with llama3 + SFT + word
    python scripts/merge_lora.py --model llama3 --paradigm sft --trigger-type word

    # Merge with custom paths
    python scripts/merge_lora.py --lora-dir /path/to/lora --base-model /path/to/model --output /path/out
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEFAULT_MODEL,
    get_model_dir,
    get_artifact_dir,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter back into base model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Model short alias (llama3/qwen2.5/mistral). "
            f"Default: {DEFAULT_MODEL}"
        )
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        default="sft",
        choices=["sft", "rlhf", "badedit"],
        help="Paradigm used for training (determines artifact path)"
    )
    parser.add_argument(
        "--trigger-type",
        type=str,
        default="word",
        choices=["word", "phrase", "long"],
        help="Trigger type used for training (determines artifact path)"
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=None,
        help="LoRA adapter directory (default: artifacts/{model}/{paradigm}/{trigger}/lora/)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path (default: resolved from --model alias)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/{model}/{paradigm}/{trigger}/merged/)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for merging (default: auto)"
    )
    return parser.parse_args()


def main():
    """Merge LoRA adapter into base model."""
    args = parse_args()

    # Resolve paths
    base_model_path = args.base_model or get_model_dir(args.model)
    lora_dir = args.lora_dir or get_artifact_dir(args.model, args.paradigm, args.trigger_type, "lora")
    output_dir = args.output_dir or get_artifact_dir(args.model, args.paradigm, args.trigger_type, "merged")

    print("=" * 50)
    print("LoRA Merge")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {lora_dir}")
    print(f"Output: {output_dir}")

    # Validate paths exist
    if not os.path.exists(base_model_path):
        print(f"Error: Base model not found: {base_model_path}")
        sys.exit(1)
    if not os.path.exists(lora_dir):
        print(f"Error: LoRA adapter not found: {lora_dir}")
        print("Please run train.py first to produce the LoRA adapter.")
        sys.exit(1)

    # Import HF deps (deferred so --help works without them)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"Error: Missing dependencies: {e}")
        print("Install transformers + peft + torch to merge LoRA.")
        sys.exit(1)

    # Load base model
    print("\n1. Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device is None else {"": args.device},
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("\n2. Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_dir)

    # Merge and unload
    print("\n3. Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    # Save merged model
    print("\n4. Saving merged model...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Merged model saved to: {output_dir}")

    print("\n" + "=" * 50)
    print("Merge completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()