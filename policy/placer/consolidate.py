class ConsolidatePlacement:
    def __init__(self, vc):
        self.name = "consolidate"
        self.vc = vc

    # 整卡正向调度
    def place(self, job):
        gpu_request = job["gpu_request"]

        while len(job["nodes"]) < job["worker_num"]:
            if gpu_request < 1.0:
                share_flag, alloc_node, alloc_gpu = self.gpu_share_select(job)
                if share_flag:
                    assert alloc_node.allocate_share_gpu(alloc_gpu, job)
                    job["nodes"].append({alloc_node.node_name: [alloc_gpu]})
                    continue

            gpu_request = int(max(job["gpu_request"], 1))
            select_flag, alloc_node = self.gpu_select(job)

            """ Placement """
            if select_flag:
                allocate_gpus = alloc_node.allocate_gpu(gpu_request, job)
                job["nodes"].append({alloc_node.node_name: allocate_gpus})
                continue
            else:
                return False
        return True

    # 碎卡抢占
    def gpu_share_select(self, job):
        job_gpu_num = job["gpu_request"]

        # 筛选可行节点，模拟调度器Filter阶段
        frag_gpus = []
        min_gpu = 1.0
        for node in self.vc.node_list:
            for i in range(node.num_gpus):
                galloc = node.node_galloc[i]
                if 0 < galloc < 1.0 - job_gpu_num:
                    if 1.0 - galloc < min_gpu:
                        frag_gpus = [[node, i, 1.0 - galloc]]
                        min_gpu = 1.0 - galloc

        # 寻找最优节点，模拟调度器Score阶段
        if len(frag_gpus) == 0:
            return False, None, None
        node, gpu = frag_gpus[0][0], frag_gpus[0][1]
        return True, node, gpu

    # 整卡抢占
    def gpu_select(self, job):
        job_gpu_num = int(max(job["gpu_request"], 1))

        # 筛选可行节点，模拟调度器Filter阶段
        avail_node_list = []
        min_gpu = 16
        for node in self.vc.node_list:
            free_gpus = node.free_gpus
            if job_gpu_num <= free_gpus:
                if free_gpus < min_gpu:
                    avail_node_list = [node]
                    min_gpu = free_gpus

        # 寻找最优节点，模拟调度器Score阶段
        if len(avail_node_list) == 0:
            return False, None
        node = avail_node_list[0]
        return True, node
