"""
Artifact path resolution and LoRA merge helpers.

    resolve_model_path         - resolve artifact path (merged/ or poisoned/) from CLI args
    auto_merge_lora_if_needed  - auto-merge LoRA adapters when merged/ is missing
    cleanup_auto_merged        - delete a model dir created by auto-merge
"""

import os
import shutil

from config import get_artifact_dir, get_model_dir, DEVICE


def resolve_model_path(args) -> str:
    """Resolve model artifact path from CLI args (merged/ for LoRA, poisoned/ for badedit)."""
    if args.model_path:
        return args.model_path
    artifact = "poisoned" if args.paradigm == "badedit" else "merged"
    return get_artifact_dir(
        args.model, dataset=args.dataset, paradigm=args.paradigm,
        trigger_type=args.trigger_type, artifact=artifact,
    )


def auto_merge_lora_if_needed(args, model_path: str) -> bool:
    """
    Auto-merge LoRA into the base model when the merged model is missing.

    Only applies to LoRA paradigms (not badedit) when --auto-merge is set.

    Returns:
        True if auto-merging was performed, False otherwise.
    """
    if args.paradigm == "badedit" or not args.auto_merge:
        return False
    if os.path.exists(model_path):
        return False

    lora_dir = get_artifact_dir(
        args.model, dataset=args.dataset, paradigm=args.paradigm,
        trigger_type=args.trigger_type, artifact="lora",
    )
    base_model_path = get_model_dir(args.model)
    if not os.path.exists(lora_dir):
        print(
            f"Warning: LoRA adapter not found at {lora_dir}. "
            "Please run train.py first."
        )
        return False

    print("\n0. Auto-merging LoRA adapter...")
    try:
        from scripts.merge_lora import merge_lora_adapter
        merge_lora_adapter(
            base_model_path=base_model_path,
            lora_dir=lora_dir,
            output_dir=model_path,
            device=DEVICE,
        )
        return True
    except Exception as e:
        print(f"Error: Auto-merge failed: {e}")
        return False


def cleanup_auto_merged(model_path: str, auto_merged: bool, keep: bool) -> None:
    """Delete the auto-merged model unless --keep-merged is set."""
    if auto_merged and not keep:
        print("\nCleaning up auto-merged model...")
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
            print(f"🗑️  Deleted merged model: {model_path}")
        else:
            print(f"Warning: Auto-merged model not found: {model_path}")