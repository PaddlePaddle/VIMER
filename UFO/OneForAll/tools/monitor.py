import math
import paddle
import numpy as np

def _alltoall(in_tensor_list, group=None, use_calc_stream=True):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    nranks = len(in_tensor_list)
    return paddle._C_ops.alltoall(in_tensor_list, 'use_calc_stream', use_calc_stream,
                              'ring_id', ring_id)

class Monitor:
    def __init__(self, tasks=29, custom_task2ranks=[8,4,4,4], dp_degree=1):
        self.tasks = tasks
        self.custom_task2ranks = custom_task2ranks
        self.dp_degree = dp_degree
        self.sub_groups, self.moe_groups, self.dp_groups, self.fake_data = self._get_group_ranks(dp_degree)
        self.sub_group, self.moe_group, self.dp_group = self._get_cur_group()
        cur_dp_rank = paddle.distributed.get_rank() // self.moe_group.nranks
        # self.sub_index = self.sub_groups[cur_dp_rank].index(self.sub_group if self.sub_group is not None else [paddle.distributed.get_rank()])
        self.rankid2taskindex = {}
        print(self.sub_groups[0])
        for taskindex, group in enumerate(self.sub_groups[cur_dp_rank]):
            for rankid in group:
                self.rankid2taskindex[rankid] = taskindex

    def _get_expert_count(self, task_id):
        """
            task_id: list of the number of each task data, example: task_id = [32, 16, 8, 8]
        """
        
        # assert task_id.numel() == self.moe_group.nranks, "the number of dataloaders must be equal moe_group's size({})".format(self.moe_group.nranks)
        #Note: each rank must be read the same number of each task data.
        local_expert_count = task_id.astype("int64")
        all_count = paddle.tile(local_expert_count, [self.moe_group.nranks, 1])
        cur_rank = paddle.distributed.get_rank()
        cur_dp_rank = paddle.distributed.get_rank() // self.moe_group.nranks

        belong_sub_group = self.sub_groups[cur_dp_rank]
        index = belong_sub_group.index(self.sub_group.ranks if self.sub_group is not None else [cur_rank])
        global_expert_count = all_count[:, index].reshape([-1])
         
        if self.custom_task2ranks[index] == 1:
            global_expert_count = paddle.to_tensor(global_expert_count).astype("int64")
        else:
            send_len = self.moe_group.nranks // self.custom_task2ranks[index]
            mask = np.zeros_like(global_expert_count)
            ss = self.sub_group.ranks.index(cur_rank)
            mask[ss * send_len : ss * send_len + send_len] = 1
            global_expert_count *= paddle.to_tensor(mask)
            global_expert_count = global_expert_count.astype("int64")
        print('-'*30)
        print('global_expert_count is ' ,global_expert_count)
        local_expert_count = _alltoall(global_expert_count, group = self.moe_group)
        print('local_expert_count is ', local_expert_count)
        return local_expert_count, global_expert_count

    
    def _get_cur_group(self, rank=None):
        cur_rank = paddle.distributed.get_rank() if rank is None else rank
        cur_dp_rank = cur_rank // len(self.moe_groups[0])

        sub_group = None
        moe_group = None
        dp_group = None
        #print(cur_rank)
        #print("before1")
        for i, sg in enumerate(self.sub_groups):
            for ssg in sg:
                if len(ssg) <= 1: continue
                #print(ssg)
                tmp = paddle.distributed.new_group(ssg)
                #paddle.device.cuda.synchronize()
                #print('after inside before1')
                #print(ssg)
                if cur_dp_rank == i and cur_rank in ssg:
                    sub_group = tmp
        #print("before2")
        for i, mg in enumerate(self.moe_groups):
            tmp = paddle.distributed.new_group(mg)
            if cur_rank in mg:
                moe_group = tmp
        #print("before3")
        for dg in self.dp_groups:
            #if len(dg) <= 1: continue
            tmp = paddle.distributed.new_group(dg)
            if cur_rank in dg:
                dp_group = tmp
        #print("before4")
        if sub_group is not None and sub_group.nranks > 1:
            assert moe_group.nranks % sub_group.nranks == 0, "error! you must be sure that moe_group's size({}) can be divided by sub_group'size({})." \
                                                          .format(moe_group.nranks, sub_group.nranks)
        return sub_group, moe_group, dp_group
    

    def _get_group_ranks(self, dp_degree=1):
        tasks = self.tasks
        custom_task2ranks = self.custom_task2ranks
        assert len(custom_task2ranks) <= tasks, "Error, the number of custom task mapping to rank must be less than {}".format(tasks)
        
        left = tasks - len(custom_task2ranks)
        
        custom_task2ranks += ([1] * left) # one moe_group
        
        if paddle.distributed.get_world_size() <= 6:
            nodes = 1
            world_size = paddle.distributed.get_world_size() // dp_degree
        else:
            nodes = math.ceil((sum(custom_task2ranks) / 6))
            world_size = nodes * 6
            world_size *= dp_degree 
        
        assert paddle.distributed.get_world_size() == world_size * dp_degree, "error! the number of nodes must be {} containing {} ranks in a dp group.".format(nodes, world_size)
        
        fake_data = world_size - sum(custom_task2ranks)
        print("\n====>Info: need {} nodes(ranks={}), need {} fake rank in a dp group.".format(nodes, world_size, fake_data))
        
        custom_task2ranks.extend([1] * fake_data)
        custom_cumsum = np.array(custom_task2ranks).cumsum()
        tmp = custom_cumsum.tolist()
        end = tmp.pop(-1)
        global_group = [0] + tmp
        sub_group = []
        for i in range(len(global_group)):
            ss = global_group[i]
            if i + 1 >= len(global_group):
                ee = end
            else:
                ee = global_group[i + 1]
            sub_group.append(list(range(ss, ee)))
        moe_group = list(range(world_size))
        moe_groups = [list(map(lambda a:a + world_size * i, moe_group)) for i in range(dp_degree)]
        dp_groups = [list(range(world_size * dp_degree))[i::world_size] for i in range(world_size)]
        sub_groups=[[list(map(lambda a: a + world_size * i, sub)) for sub in sub_group] for i in range(dp_degree) ]

        print("==>monitor sub_groups: ", sub_groups)
        print("==>monitor moe_groups: ", moe_groups)
        print("==>monitor dp_groups: ", dp_groups)

        return sub_groups, moe_groups, dp_groups, fake_data

if __name__ == "__main__":
    m = Monitor(dp_degree=2)
    #sub_group, moe_group, dp_group = m.get_cur_group()
    #print(sub_group, "\n", moe_group,"\n", dp_group)
