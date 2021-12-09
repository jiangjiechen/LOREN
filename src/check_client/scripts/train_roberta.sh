#!/usr/bin/env bash

MODEL_TYPE=roberta
MODEL_NAME_OR_PATH=roberta-large
VERSION=v5
MAX_NUM_QUESTIONS=8

MAX_SEQ1_LENGTH=256
MAX_SEQ2_LENGTH=128
CAND_K=3
LAMBDA=${1:-0.5}
PRIOR=${2:-nli}
MASK=${3:-0.0}
echo "lambda = $LAMBDA, prior = $PRIOR, mask = $MASK"

DATA_DIR=$PJ_HOME/data/fact_checking/${VERSION}
OUTPUT_DIR=$PJ_HOME/models/fact_checking/${VERSION}_${MODEL_NAME_OR_PATH}/${VERSION}_${MODEL_NAME_OR_PATH}_AAAI_K${CAND_K}_${PRIOR}_m${MASK}_l${LAMBDA}
NUM_TRAIN_EPOCH=7
GRADIENT_ACCUMULATION_STEPS=2
PER_GPU_TRAIN_BATCH_SIZE=8 # 4546
PER_GPU_EVAL_BATCH_SIZE=16
LOGGING_STEPS=200
SAVE_STEPS=200


python3 train.py \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_type ${MODEL_TYPE} \
  --model_name_or_path ${MODEL_NAME_OR_PATH} \
  --max_seq1_length ${MAX_SEQ1_LENGTH} \
  --max_seq2_length ${MAX_SEQ2_LENGTH} \
  --max_num_questions ${MAX_NUM_QUESTIONS} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --learning_rate 1e-5 \
  --num_train_epochs ${NUM_TRAIN_EPOCH} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --per_gpu_train_batch_size ${PER_GPU_TRAIN_BATCH_SIZE} \
  --per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH_SIZE} \
  --logging_steps ${LOGGING_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --cand_k ${CAND_K} \
  --logic_lambda ${LAMBDA} \
  --prior ${PRIOR} \
  --overwrite_output_dir \
  --temperature 1.0 \
  --mask_rate ${MASK}

python3 cjjpy.py --lark "$OUTPUT_DIR fact checking training completed"
