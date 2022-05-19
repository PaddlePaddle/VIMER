#!/bin/bash

# extract features for eval
test_list="./Product1M/product1m_test_ossurl_v2.txt"  
image_root="./Product1M/product1m_test_ossurl_v2" 
test_prefix="product1m_test"


CUDA_VISIBLE_DEVICES=0 python3 -u eval_ums.py \
    --model_type 'VL' \
    --text_file $test_list \
    --image_root $image_root \
    --save_prefix $test_prefix \
    --resume_file $1 \
    --pre_training_model 'ITCL' \
    --image_model_name 'vit_deit_base_patch16_384' \
    --text_model_dir './configs/' \
    --dataset 'Product1M' \
    --image_size 384 \
    --projection_dim 512 \
    --image_embed_dim 768 \
    --text_embed_dim 768 \
    --max_seq_length 40 \
    --eval_batch_size 30 \
    --feature_type "itcl" \
    --num_workers 8

# extract features for eval
test_list="./Product1M/product1m_gallery_ossurl_v2.txt"  
image_root="./Product1M/product1m_gallery_ossurl_v2" 
test_prefix="product1m_gallery"


CUDA_VISIBLE_DEVICES=0 python3 -u eval_ums.py \
    --model_type 'VL' \
    --text_file $test_list \
    --image_root $image_root \
    --save_prefix $test_prefix \
    --resume_file $1 \
    --pre_training_model 'ITCL' \
    --image_model_name 'vit_deit_base_patch16_384' \
    --text_model_dir './configs/' \
    --dataset 'Product1M' \
    --image_size 384 \
    --projection_dim 512 \
    --image_embed_dim 768 \
    --text_embed_dim 768 \
    --max_seq_length 40 \
    --eval_batch_size 30 \
    --feature_type "itcl" \
    --num_workers 8

python3 eval_scripts/get_recall_product1m_instance.py $1 

