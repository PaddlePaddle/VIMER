#!/bin/bash

# extract features for eval
test_list="../Product1M/product1m_test_ossurl_v2.txt"  
image_root="../Product1M/product1m_test_ossurl_v2/" 
test_prefix="product1m_test"
ocr_path="../Product1M/ocr_results_filter_chinese_overlap/product1m_test_ossurl_v2/"
return_ocr_pos=1


CUDA_VISIBLE_DEVICES=0 python3 -u eval_ums.py \
    --model_type 'VL' \
    --text_file $test_list \
    --image_root $image_root \
    --save_prefix $test_prefix \
    --ocr_path $ocr_path \
    --return_ocr_pos $return_ocr_pos \
    --resume_file $1 \
    --pre_training_model 'ViSTA_BASE' \
    --fusion_depth 2 \
    --image_model_name 'fusion_base_patch16_224' \
    --text_model_dir './config_p1m/' \
    --scene_text_model_dir './config_p1m/' \
    --dataset 'Product1M' \
    --image_size 224 \
    --projection_dim 512 \
    --image_embed_dim 768 \
    --text_embed_dim 768 \
    --scene_text_embed_dim 768 \
    --max_seq_length 40 \
    --max_scene_text_length 30 \
    --eval_batch_size 30 \
    --feature_type "both" \
    --num_workers 8

# extract features for eval
test_list="../Product1M/product1m_gallery_ossurl_v2.txt"  
image_root="../Product1M/product1m_gallery_ossurl_v2/" 
test_prefix="product1m_gallery"
ocr_path="../Product1M/ocr_results_filter_chinese_overlap/product1m_gallery_ossurl_v2/"
return_ocr_pos=1


CUDA_VISIBLE_DEVICES=0 python3 -u eval_ums.py \
    --model_type 'VL' \
    --text_file $test_list \
    --image_root $image_root \
    --save_prefix $test_prefix \
    --ocr_path $ocr_path \
    --return_ocr_pos $return_ocr_pos \
    --resume_file $1 \
    --pre_training_model 'ViSTA_BASE' \
    --fusion_depth 2 \
    --image_model_name 'fusion_base_patch16_224' \
    --text_model_dir './config_p1m/' \
    --scene_text_model_dir './config_p1m/' \
    --dataset 'Product1M' \
    --image_size 224 \
    --projection_dim 512 \
    --image_embed_dim 768 \
    --text_embed_dim 768 \
    --scene_text_embed_dim 768 \
    --max_seq_length 40 \
    --max_scene_text_length 30 \
    --eval_batch_size 30 \
    --feature_type "both" \
    --num_workers 8

python3 eval_scripts/get_recall_product1m_instance.py $1 

