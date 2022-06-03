import paddle
import os
import argparse
from collections import OrderedDict
import logging 
import numpy as np

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='For ViT')
parser.add_argument('--paddle_model_path', type=str)
parser.add_argument('--task_name', type=str)
args = parser.parse_args()

# --paddle_model_path 'UFO_2.0_17B_release.pdmodel'

task_names=['face', 'person', 'veri', 'sop', 'task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7',
                    'task8', 'task9', 'task0', 'task10', 'task11', 
                    # 'task12', 'task13', 'task14', 'task15', 
                    'task16',
                    'task17', 'task18', 'task19', 'task20', 
                    # 'task21', 
                    'task22', 'task23', 'task24', 'dmlab', 'retinopathy']
taskname2taskid = {task_name: idx for idx, task_name in enumerate(task_names)}
state_dict = paddle.load(args.paddle_model_path)
state_dict_new = {}
for k, v in state_dict.items():
    if 'pos_embeds' in k:
        taskid = int(k.split('.')[2])
        if taskid == taskname2taskid[args.task_name]:
            state_dict_new['backbone.pos_embed'] = v
    elif 'cls_tokens' in k:
        taskid = int(k.split('.')[2])
        if taskid == taskname2taskid[args.task_name]:
            state_dict_new['backbone.cls_token'] = v
    elif 'patch_embeds' in k:
        task_name = k.split('.')[2]
        if task_name == args.task_name:
            if 'weight' in k:
                    state_dict_new['backbone.patch_embed.proj.weight'] = v
            elif 'bias' in k:
                state_dict_new['backbone.patch_embed.proj.bias'] = v
            else:
                raise NotImplementedError()
    elif 'head' in k:
        task_name = k.split('.')[1]
        if task_name == args.task_name:
            state_dict_new[k] = v

for block_idx in range(32):
    expert_weight = state_dict['backbone.blocks.{}.mlp.policy'.format(block_idx)]
    for k_new in ['backbone.blocks.{}.mlp.fc1.weight'.format(block_idx), 'backbone.blocks.{}.mlp.fc1.bias'.format(block_idx), 'backbone.blocks.{}.mlp.fc2.weight'.format(block_idx), 'backbone.blocks.{}.mlp.fc2.bias'.format(block_idx)]:
        if expert_weight[taskname2taskid[args.task_name]][0] > expert_weight[taskname2taskid[args.task_name]][1]:
            state_dict_new[k_new] = state_dict[k_new.replace('mlp', 'mlp.common_expert')]
        else:
            state_dict_new[k_new] = state_dict[k_new.replace('mlp', 'mlp.specific_experts.{}'.format(args.task_name))]
    for k_new in ['backbone.blocks.{}.norm2.weight'.format(block_idx), 'backbone.blocks.{}.norm2.bias'.format(block_idx)]:
        if expert_weight[taskname2taskid[args.task_name]][0] > expert_weight[taskname2taskid[args.task_name]][1]:
            state_dict_new[k_new] = state_dict[k_new.replace('norm2', 'mlp.common_norm')]
        else:
            state_dict_new[k_new] = state_dict[k_new.replace('norm2', 'mlp.specific_norms.' + args.task_name)]                

    expert_weight = state_dict['backbone.blocks.{}.attn.policy'.format(block_idx)]
    for k_new in ['backbone.blocks.{}.attn.qkv.weight'.format(block_idx), 'backbone.blocks.{}.attn.qkv.bias'.format(block_idx), 'backbone.blocks.{}.attn.proj.weight'.format(block_idx), 'backbone.blocks.{}.attn.proj.bias'.format(block_idx)]:
        if expert_weight[taskname2taskid[args.task_name]][0] > expert_weight[taskname2taskid[args.task_name]][1]:
            state_dict_new[k_new] = state_dict[k_new.replace('attn', 'attn.common_expert')]
        else:
            state_dict_new[k_new] = state_dict[k_new.replace('attn', 'attn.specific_experts.{}'.format(args.task_name))]
    for k_new in ['backbone.blocks.{}.norm1.weight'.format(block_idx), 'backbone.blocks.{}.norm1.bias'.format(block_idx)]:
        if expert_weight[taskname2taskid[args.task_name]][0] > expert_weight[taskname2taskid[args.task_name]][1]:
            state_dict_new[k_new] = state_dict[k_new.replace('norm1', 'attn.common_norm')]
        else:
            state_dict_new[k_new] = state_dict[k_new.replace('norm1', 'attn.specific_norms.' + args.task_name)]

expert_weight = state_dict['backbone.norm.policy']
for k_new in ['backbone.norm.weight', 'backbone.norm.bias']:
    if expert_weight[taskname2taskid[args.task_name]][0] > expert_weight[taskname2taskid[args.task_name]][1]:
        state_dict_new[k_new] = state_dict[k_new.replace('norm', 'norm.common_expert')]
    else:
        state_dict_new[k_new] = state_dict[k_new.replace('norm', 'norm.specific_experts.{}'.format(args.task_name))]
print(state_dict_new.keys())
paddle.save(state_dict_new, args.paddle_model_path + '.to{}'.format(args.task_name))
