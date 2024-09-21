#!/bin/bash

# get the paper name
paper_name=$1

if [ -z "$paper_name" ]; then
  echo "please offer the paper name"
  exit 1
fi

# Check if the file exists
if [ ! -f "./template/${paper_name}_output.jsonl" ]; then
  echo "File not found, running summarize.py"
  python3 "summarize.py" "$paper_name"
else
  echo "File exists, skipping summarize.py"
fi

# Execute attack.py
python3 "attack.py" "$paper_name"
