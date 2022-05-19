#!/bin/bash

# extract features for eval
test_list="./data/CTC_1K_list_ocr_release.txt"  
ocr_feature_files="None"
return_ocr_feature=3 #0:no ocr  1: return pad mask  2:retrun ocr feature 3:read ocr text and return ocr feature
ocr_pos_path="./data/ctc_ocr/"
return_ocr_2D_pos=1
image_root="./data" # ./COCO-CTC
test_prefix="ctc_1k_scene_text" 


CUDA_VISIBLE_DEVICES=0 python3 -u eval_vista.py \
    --text_file $test_list \
    --ocr_feature_files $ocr_feature_files \
    --return_ocr_feature $return_ocr_feature \
    --ocr_pos_path $ocr_pos_path \
    --ocr_pos_path "./data/ctc_ocr/" \
    --return_ocr_2D_pos $return_ocr_2D_pos \
    --image_root $image_root \
    --save_prefix $test_prefix \
    --resume_file $1 \
    --pre_training_model 'ViSTA' \
    --fusion_depth 4 \
    --image_model_name 'fusion_small_patch16_224' \
    --text_model_dir './configs/' \
    --scene_text_model_dir './configs/' \
    --image_size 224 \
    --projection_dim 256 \
    --image_embed_dim 384 \
    --text_embed_dim 256 \
    --scene_text_embed_dim 256 \
    --max_seq_length 40 \
    --max_scene_text_length 30 \
    --eval_batch_size 128 \
    --feature_type "both" \
    --num_workers 8

python3 eval_scripts/get_recall.py $test_prefix $1
