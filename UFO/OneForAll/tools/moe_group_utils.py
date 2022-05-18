import paddle
import numpy as np

def get_moe_group(dp_degree):
    #构建 moe group
    world_size = paddle.distributed.get_world_size()
    # dp_degree = cfg.train.dp_degree
    assert world_size % dp_degree == 0, "Error! be sure that world_size '%' dp_degree == 0"
    print("==================trainer info===================")
    ranks = list(range(world_size))
    group_len = len(ranks) // dp_degree
    global_group = paddle.distributed.new_group(ranks)
    cur_rank = paddle.distributed.get_rank()
    print("==>cur_rank: ", cur_rank)
    print("==>global_group: ", global_group)
    dp_groups = [ranks[i::group_len] for i in range(group_len)]
    moe_groups = np.split(np.array(ranks), dp_degree)
    cur_dp_group = None
    for dp in dp_groups:
        print("==>", dp)
        tmp = paddle.distributed.new_group(dp)
        if cur_rank in dp:
            cur_dp_group = tmp

    cur_moe_group = None
    for mp in moe_groups:
        print("==>", mp)
        tmp = paddle.distributed.new_group(mp.tolist())
        if cur_rank in mp.tolist():
            cur_moe_group = tmp 
    print("==>cur_dp_group: ", cur_dp_group)
    print("==>cur_moe_group: ", cur_moe_group)
    print("======================================")
    return cur_moe_group


def get_dp_group(dp_degree):
    #构建 moe group
    world_size = paddle.distributed.get_world_size()
    # dp_degree = cfg.train.dp_degree
    assert world_size % dp_degree == 0, "Error! be sure that world_size '%' dp_degree == 0"
    print("==================trainer info===================")
    ranks = list(range(world_size))
    group_len = len(ranks) // dp_degree
    global_group = paddle.distributed.new_group(ranks)
    cur_rank = paddle.distributed.get_rank()
    print("==>cur_rank: ", cur_rank)
    print("==>global_group: ", global_group)
    dp_groups = [ranks[i::group_len] for i in range(group_len)]
    moe_groups = np.split(np.array(ranks), dp_degree)
    cur_dp_group = None
    for dp in dp_groups:
        print("==>", dp)
        tmp = paddle.distributed.new_group(dp)
        if cur_rank in dp:
            cur_dp_group = tmp

    cur_moe_group = None
    for mp in moe_groups:
        print("==>", mp)
        tmp = paddle.distributed.new_group(mp.tolist())
        if cur_rank in mp.tolist():
            cur_moe_group = tmp 
    print("==>cur_dp_group: ", cur_dp_group)
    print("==>cur_moe_group: ", cur_moe_group)
    print("======================================")
    return cur_dp_group