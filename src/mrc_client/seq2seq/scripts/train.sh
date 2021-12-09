#!/usr/bin/env bash

echo "--data_dir --model_name_or_path=facebook/bart-base --output_dir"

python3 finetune.py \
  --gpus 8 \
  --do_train \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 20 \
  --max_source_length 400 \
  --max_target_length 15 \
  --val_max_target_length 15 \
  --test_max_target_length 15 \
  --min_target_length 1 \
  --val_check_interval 0.5 \
  --n_val -1 \
  --save_top_k 5 \
  --logger_name wandb \
  --overwrite_output_dir \
  "$@"

python3 cjjpy.py --lark "training mrc-seq2seq completed"
