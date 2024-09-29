#!/bin/bash

# 定义一个包含paper_name的列表
paper_names=("autodan")  # 你可以在这里添加更多的paper name

# 设置默认值，如果没有提供这些参数，则使用默认值
title=${1:-64}
author=${2:-64}
attack_methods=${3:-256}
introduction_to_the_mechanism_of_success=${4:-512}
related_work=${5:-128}

# 循环处理每个paper_name
for paper_name in "${paper_names[@]}"; do

  echo "Processing paper: $paper_name"
  
  # 定义文件路径
  file_path="../template/${paper_name}_${title}_${author}_${attack_methods}_${introduction_to_the_mechanism_of_success}_${related_work}.jsonl"

  # 检查文件是否存在，如果不存在则运行 summarize.py
  if [ ! -f "$file_path" ]; then
    echo "File not found for $paper_name, running summarize.py"
    python3 "../summarize.py" "$paper_name" "$title" "$author" "$attack_methods" "$introduction_to_the_mechanism_of_success" "$related_work"
  else
    echo "File exists for $paper_name, skipping summarize.py."
  fi

  # 无论文件是否存在，都运行 ablation.py
  python3 "time_step_analysis.py" "$paper_name" "$title" "$author" "$attack_methods" "$introduction_to_the_mechanism_of_success" "$related_work"

done
