#!/bin/bash

# extract features for eval
test_list="./data/CTC_1K_list_ocr_release.txt"  
image_root="./data/" 
test_prefix="ctc_1k"
ocr_path="./data/ctc_ocr"
return_ocr_pos=1


CUDA_VISIBLE_DEVICES=0 python3 -u eval_ums.py \
    --model_type 'VL' \
    --text_file $test_list \
    --image_root $image_root \
    --save_prefix $test_prefix \
    --ocr_path $ocr_path \
    --return_ocr_pos $return_ocr_pos \
    --resume_file $1 \
    --pre_training_model 'ViSTA_BASE_SHARE' \
    --fusion_depth 2 \
    --image_model_name 'fusion_base_patch16_224' \
    --text_model_dir './config_ctc/' \
    --scene_text_model_dir './config_ctc/' \
    --dataset 'CTC' \
    --image_size 224 \
    --projection_dim 512 \
    --image_embed_dim 768 \
    --text_embed_dim 768 \
    --scene_text_embed_dim 768 \
    --max_seq_length 40 \
    --max_scene_text_length 30 \
    --eval_batch_size 32 \
    --feature_type "both" \
    --num_workers 8

python3 eval_scripts/get_recall.py $test_prefix $1

