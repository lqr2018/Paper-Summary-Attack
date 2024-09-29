#!/bin/bash

# get the paper name
paper_name=$1
title=${2:-64}
author=${3:-64}
attack_methods=${4:-256}
introduction_to_the_mechanism_of_success=${5:-512}
related_work=${6:-128}
if [ -z "$paper_name" ]; then
  echo "please offer the paper name"
  exit 1
fi


if [ -z "$title" ]; then
  echo "please offer the title generate token"
  exit 1
fi

if [ -z "$author" ]; then
  echo "please offer the Author generate token"
  exit 1
fi
if [ -z "$attack_methods" ]; then
  echo "please offer the attack_methods generate token"
  exit 1
fi
if [ -z "$introduction_to_the_mechanism_of_success" ]; then
  echo "please offer the Introduction_to_the_Mechanism_of_Success generate token"
  exit 1
fi
if [ -z "$related_work" ]; then
  echo "please offer the related_work  generate token"
  exit 1
fi

# Check if the file exists
if [ ! -f "../template/${paper_name}_${title}_${author}_${attack_methods}_${introduction_to_the_mechanism_of_success}_${related_work}.jsonl" ]; then
  echo "File not found, running summarize.py"
  python3 "../summarize.py" "$paper_name" "$title" "$author" "$attack_methods" "$introduction_to_the_mechanism_of_success" "$related_work"
else
  echo "File exists, skipping summarize.py"
fi

# Execute attack.py
python3 "../attack.py" "$paper_name" "$title" "$author" "$attack_methods" "$introduction_to_the_mechanism_of_success" "$related_work"
