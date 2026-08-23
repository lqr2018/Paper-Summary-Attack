"""
Evaluation result persistence.

    save_eval_results - write per-run JSON (latest) + cumulative CSV summary.
"""

import csv
import datetime
import json
import os
from typing import Dict, Any

from config import RESULTS_DIR


def save_eval_results(args, summary: Dict[str, Any]) -> None:
    """
    Save evaluation results:
      ① JSON (latest, overwrite) -> outputs/results/{model}/{dataset}/{paradigm}/{trigger}/
      ② CSV (append, cumulative) -> outputs/results/eval_summary.csv
    """
    result_dir = os.path.join(
        RESULTS_DIR, args.model, args.dataset, args.paradigm, args.trigger_type
    )
    os.makedirs(result_dir, exist_ok=True)

    # ① JSON: 每次覆盖,保存最新完整结果
    json_path = os.path.join(result_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Evaluation results saved to: {json_path}")

    # ② CSV: 汇总文件追加一行(按时间戳),便于累积对比
    # 保留 generation_asr 列与 evaluate_dpo.py 对齐(SFT/BadEdit 该列留空)
    csv_path = os.path.join(RESULTS_DIR, "eval_summary.csv")
    fieldnames = [
        "timestamp", "dataset", "model", "paradigm", "trigger_type",
        "clean_accuracy", "clean_total", "clean_correct",
        "poisoned_accuracy", "poisoned_total", "poisoned_correct",
        "generation_asr",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "model": args.model,
            "paradigm": args.paradigm,
            "trigger_type": args.trigger_type,
            "clean_accuracy": summary["clean_accuracy"],
            "clean_total": summary["clean_total"],
            "clean_correct": summary["clean_correct"],
            "poisoned_accuracy": summary["poisoned_accuracy"],
            "poisoned_total": summary["poisoned_total"],
            "poisoned_correct": summary["poisoned_correct"],
            "generation_asr": None,
        })
    print(f"📄 Evaluation summary appended to: {csv_path}")