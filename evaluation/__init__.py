"""
Reusable evaluation helpers shared across scripts (evaluate.py, evaluate_dpo.py).

Submodules:
    evaluation.model     - model loading, prompt formatting, prediction, label normalization
    evaluation.artifacts - artifact path resolution, LoRA auto-merge, merged-model cleanup
    evaluation.results   - persist evaluation summaries (JSON + CSV)
"""

from .model import (
    load_model,
    format_prompt,
    predict,
    extract_label,
)
from .artifacts import (
    resolve_model_path,
    auto_merge_lora_if_needed,
    cleanup_auto_merged,
)
from .results import save_eval_results

__all__ = [
    "load_model",
    "format_prompt",
    "predict",
    "extract_label",
    "resolve_model_path",
    "auto_merge_lora_if_needed",
    "cleanup_auto_merged",
    "save_eval_results",
]