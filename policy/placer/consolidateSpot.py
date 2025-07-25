class ConsolidateSpotPlacement:
    def __init__(self, vc):
        self.name = "consolidate_spot"
        self.vc = vc

    # 整卡正向调度
    def consolidate_select(self, job_gpu_num, job_type):
        scheduler_score_dict = self.vc.scheduler_score_dict(job_gpu_num, job_type)
        if len(scheduler_score_dict) == 0:
            return False, None
        if self.vc.eviction_score is None:
            nodes = sorted(scheduler_score_dict.keys(), key=lambda x: scheduler_score_dict[x], reverse=False)
        elif job_type == "Spot":
            nodes = sorted(scheduler_score_dict.keys(), key=lambda x: (scheduler_score_dict[x],
                                                                       -self.vc.eviction_score[x]), reverse=False)
        else:
            nodes = sorted(scheduler_score_dict.keys(), key=lambda x: (scheduler_score_dict[x],
                                                                       self.vc.eviction_score[x]), reverse=False)
        return True, self.vc.get_node(nodes[0])

    # 碎卡正向调度
    def gpu_share_select(self, job_gpu_num):
        frag_gpus = self.vc.frag_gpu_dict(job_gpu_num=job_gpu_num)
        for k, v in frag_gpus.items():
            if v >= job_gpu_num:
                return True, k[0], k[1]
        return False, None, None

    def place(self, job):
        gpu_request = job["gpu_request"]
        job_type = job["type"]

        while len(job["nodes"]) < job["worker_num"]:
            if gpu_request < 1.0:
                share_flag, alloc_node, alloc_gpu = self.gpu_share_select(gpu_request)
                if share_flag:
                    assert alloc_node.allocate_share_gpu(alloc_gpu, job)
                    job["nodes"].append({alloc_node.node_name: [alloc_gpu]})
                    continue

            gpu_request = int(max(job["gpu_request"], 1))
            select_flag, alloc_node = self.consolidate_select(gpu_request, job_type)

            """ Placement """
            if select_flag:
                allocate_gpus = alloc_node.allocate_gpu(gpu_request, job)
                job["nodes"].append({alloc_node.node_name: allocate_gpus})
                continue
            else:
                return False
        return True

    # 整卡抢占
    def select_with_preemption(self, job_gpu_num):
        guar_avail_nodes = self.vc.avail_node_list_with_preemption(job_gpu_num=job_gpu_num)
        if len(guar_avail_nodes) == 0:
            return False, None
        if self.vc.eviction_score == None:
            nodes = sorted(guar_avail_nodes, key=lambda x: x.free_gpus, reverse=False)
        else:
            nodes = sorted(guar_avail_nodes, key=lambda x: [x.free_gpus, self.vc.eviction_score[x.node_name]],
                           reverse=False)
        return True, nodes[0]

    # 碎卡抢占
    def gpu_share_select_with_preemption(self, job_gpu_num):
        guar_frag_gpus = self.vc.frag_gpu_dict_with_preemption(job_gpu_num=job_gpu_num)
        for k, v in guar_frag_gpus.items():
            if v >= job_gpu_num:
                return True, k[0], k[1]
        return False, None, None

    def place_with_preemption(self, job):
        gpu_request = job["gpu_request"]

        if gpu_request < 1.0:
            share_flag, alloc_node, alloc_gpu = self.gpu_share_select_with_preemption(gpu_request)
            if share_flag:
                preempt_spots = alloc_node.select_preempt_spots(alloc_gpu, job["gpu_request"])
                for spot in preempt_spots:
                    self.vc.release_resource(spot)

                assert alloc_node.allocate_share_gpu(alloc_gpu, job)
                job["nodes"].append({alloc_node.node_name: [alloc_gpu]})
                return True, preempt_spots

        gpu_request = int(max(job["gpu_request"], 1))
        select_flag, alloc_nodes = self.select_with_preemption(gpu_request)

        if select_flag:
            preempt_spots = []
            for alloc_node in alloc_nodes:
                alloc_gpus, node_preempt_spots = alloc_node.select_gpus_with_preemption(gpu_request)
                for spot in preempt_spots:
                    self.vc.release_resource(spot)

                assert alloc_node.allocate_gpus_with_preemption(alloc_gpus, job)
                job["nodes"].append({alloc_node.node_name: alloc_gpus})
                preempt_spots.extend(preempt_spots)
            return True, preempt_spots

        return False, None

    # spot 调度预览，回答该vc最多还可以提供多少有保障的Spot容器
    def preview_spot(self, job, max_num=1):
        gpu_request = job["gpu_request"]
        limit = min(max_num, job["worker_num"])

        count = 0
        if gpu_request < 1.0:
            frag_gpus = self.vc.frag_gpu_dict(job_gpu_num=gpu_request)
            for k, v in frag_gpus.items():
                if v >= gpu_request:
                    count += v // gpu_request
                    if count >= limit:
                        break

        for node in self.vc.node_list:
            free = node.free_gpus
            if free >= gpu_request:
                if gpu_request >= 1.0:
                    count += free // gpu_request
                else:
                    count += (1 // gpu_request) * free
                if count >= limit:
                    break

        return min(count, limit)

