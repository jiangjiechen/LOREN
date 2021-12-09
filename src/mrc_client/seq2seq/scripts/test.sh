#!/usr/bin/env bash

echo "--input_path --model_name --save_path --reference_path"

python3 run_eval.py \
  --max_length 15 \
  --min_length 1 \
  "$@"

