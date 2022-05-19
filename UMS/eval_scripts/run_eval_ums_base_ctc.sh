#!/bin/bash

# extract features for eval
test_list="./data/CTC_1k_test_list.txt"  
image_root="./data" 
test_prefix="ctc_1k"


CUDA_VISIBLE_DEVICES=0 python3 -u eval_ums.py \
    --model_type 'VL' \
    --text_file $test_list \
    --image_root $image_root \
    --save_prefix $test_prefix \
    --resume_file $1 \
    --pre_training_model 'ITCL' \
    --image_model_name 'vit_base_patch16_384' \
    --text_model_dir './configs/' \
    --dataset 'CTC' \
    --image_size 384 \
    --projection_dim 512 \
    --image_embed_dim 768 \
    --text_embed_dim 768 \
    --max_seq_length 40 \
    --eval_batch_size 30 \
    --feature_type "itcl" \
    --num_workers 8

python3 eval_scripts/get_recall.py $test_prefix $1

