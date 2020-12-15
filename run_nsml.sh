#!/usr/bin/env bash

nsml run \
  -m "+ pretrain varied mix qa for 3 epochs" \
  -d korquad-open-ldbd3 \
  -g 2 \
  -c 1 \
  -e run_squad.py \
  -a "--model_type electra
    --model_name_or_path bert-base-multilingual-cased
    --mix_qa
    --load_cache
    --load_model_session kaist007/korquad-open-ldbd3/210
    --load_model_checkpoint electra_best
    --cached_session_pretrain kaist007/korquad-open-ldbd3/369
    --cached_session_pretrain_qa kaist007/korquad-open-ldbd3/313
    --cached_session_dev kaist007/korquad-open-ldbd3/151
    --cached_session_train kaist007/korquad-open-ldbd3/151
    --do_pretrain_qa
    --do_train
    --do_eval
    --data_dir train
    --num_train_epochs 3
    --per_gpu_train_batch_size 24
    --per_gpu_eval_batch_size 24 
    --output_dir output
    --verbose_logging
    --overwrite_output_dir
    --version_2_with_negative"


