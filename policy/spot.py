import pandas as pd

from .policy import Policy
from estimator.gpu_request_estimator import GPURequestEstimator


class SpotScheduler(Policy):
    def __init__(self, trace, cluster, args, log_dir, logger):
        super(SpotScheduler, self).__init__(trace, cluster, args, log_dir, logger)
        self._name = "spot_scheduler"
        self.estimator = GPURequestEstimator(args)
        self.spot_failed, self.spot_succeed = self.init_count_dict(), self.init_count_dict()  # Spot的保障周期成功/失败率

    def simulate(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:

            """0. Update Spot Quota"""
            if self.time % 300 == 0:
                if self.args.log_range[0] <= self.time <= self.args.log_range[1]:
                    now_timestamp = pd.Timestamp(self.args.trace_range[0]) + pd.Timedelta(seconds=self.time)
                    self.logger.info(
                        f"{self.cluster_name} | Simulation Time: {now_timestamp}"
                    )
                    org_request_df = self.recorder.agg_seq_recorder()
                    if self.time % 86400 == 0:
                        self.estimator.train(org_request_df, now_timestamp)
                    self.update_spot_quota(org_request_df, now_timestamp)
                for vc_name in self.guar_que_list:
                    vc = self.cluster.vc_dict[vc_name]
                    vc.update_eviction_score(self.time)

            """1. Check & Release End Jobs"""
            for vc_name in self.run_list:
                run_ls = self.run_list[vc_name].copy()  # Avoid list.remove() issue
                vc = self.cluster.vc_dict[vc_name]
                for job in run_ls:
                    if job["type"] == "Spot" and self.time - job.get_ckpt_time() >= job["spot_duration"]:  # 达保
                        self.record_spot_succeed(job)
                        job.set_ckpt_time(self.time)

                    if job["remain"] == 0:
                        job["status"] = "end"
                        job["end_time"] = self.time
                        self.end_job_num += 1
                        assert vc.release_resource(job)
                        self.run_list[vc_name].remove(job)
                        self.gpu_request_dict[vc_name][job["type"]] -= job["gpu_request"]
                        if job["type"] == "Spot":
                            self.record_spot_succeed(job)
                        else:
                            self.guar_request_dict[job["org_key"]] -= job["gpu_request"]
                    else:
                        job["remain"] -= 1

            """2. Allocate New / Pending Jobs"""
            # New Job
            for idx in range(prev_index, self.total_job_num):
                job = self.trace.job_list[idx]
                if job["submit_time"] <= self.time:
                    self.pend_job(job)
                    prev_index = idx + 1
                elif job["submit_time"] > self.time:
                    prev_index = idx
                    break

            # Pend Job
            # NOTE: Sort by submit time -- FIFO

            for vc in self.guar_que_list:
                self.guar_que_list[vc].sort(key=lambda x: (-x.__getitem__("gpu_request"),
                                                           x.__getitem__("submit_time")))
                que_ls = self.guar_que_list[vc].copy()  # Avoid list.remove() issue
                failed_gpu_num = 16

                for job in que_ls:
                    if job["gpu_request"] >= failed_gpu_num:
                        continue
                    if self.job_placer(job):
                        self.run_guar(job)
                    elif self.time >= self.args.log_range[0]:
                        place_flag, preempt_jobs = self.place_with_preemption(job)
                        if place_flag:
                            self.run_guar(job)
                            self.preempt(job, preempt_jobs)
                        else:
                            failed_gpu_num = job["gpu_request"]
                    else:
                        failed_gpu_num = job["gpu_request"]

            for key in self.spot_que_list:
                vc = self.cluster.vc_dict[key]
                self.spot_que_list[key].sort(key=lambda x: (-x.__getitem__("gpu_request"),
                                                  x.__getitem__("worker_num"),
                                                  x.__getitem__("submit_time")))
                spot_que_ls = self.spot_que_list[key].copy()
                failed_gpu_num = 16

                # candidate_vcs = self.cluster.spot_vc[key]
                for job in spot_que_ls:
                    if job["gpu_request"] >= failed_gpu_num:
                        continue
                    if ((self.time > self.args.log_range[1]) or
                            ((self.args.log_range[0] <= self.time <= self.args.log_range[1]) and
                            (self.gpu_request_dict[key]["Spot"] + job["gpu_request"] * job["worker_num"] <=
                             self.spot_quota[key]["quota"][job["guarantee_request"]]))):
                        if self.job_placer(job):
                            self.run_spot(job)
                        else:
                            vc.release_resource(job)
                            job.update({"nodes": []})
                            failed_gpu_num = job["gpu_request"]

            """3. Log & Result Recorder"""
            # Sample Cluster State Every Minute
            if self.time % 60 == 0:
                self.recorder.update_seq_recorder()

            if self.time % 86400 == 0:
                self.runtime_log()

                if self.args.log_range[0] <= self.time:
                    self.spot_runtime_log()

            if self.args.log_range[0] <= self.time <= self.args.log_range[1]:
                self.update_guar_gpu_time()

            self.time += 1

        self.recorder.log_recorder(self._name, self.args.log_range, spot_quota_record=self.spot_quota_record)
        # self.spot_scheduler_log(log_range)

    def pend_job(self, job):
        job["status"] = "pend"
        if job["type"] == "Spot":
            # key = job["gpu_model"] + '&&' + job["cluster_name"]
            # self.spot_que_list[key].append(job)
            job.set_preempt_time(self.time)
            self.spot_que_list[job["vc_name"]].append(job)
        else:
            self.guar_que_list[job["vc_name"]].append(job)

    def run_spot(self, job):
        job["start_time"] = self.time
        job.set_ckpt_time(self.time)
        if job["preempt_times"] == 0:
            job["queue"] = self.time - job["submit_time"]
        else:
            job["queue"] = job["queue"] + (self.time - job.get_preempt_time())
        job["status"] = "run"
        if job["type"] == "Spot":
            job.update({"spot_duration": 3600})
            self.spot_que_list[job["vc_name"]].remove(job)
        else:
            raise ValueError
        self.run_list[job["vc_name"]].append(job)
        self.gpu_request_dict[job["vc_name"]][job["type"]] += job["gpu_request"]

    def preempt(self, job, preempt_jobs):
        vc_name = job["vc_name"]
        for preempt_job in preempt_jobs:
            preempt_job.update({"nodes": []})
            if preempt_job["status"] == "run":
                preempt_job["remain"] = (preempt_job["remain"] + self.preempt_overhead(preempt_job) +
                                         (self.time - preempt_job["start_time"]) % self.args.ckpt_interval)

                self.run_list[vc_name].remove(preempt_job)
                self.gpu_request_dict[vc_name][preempt_job["type"]] -= preempt_job["gpu_request"]

                if preempt_job["type"] == "Spot" and preempt_job["spot_duration"] > 0:
                    self.record_spot_failed(vc_name, preempt_job)
                self.pend_job(preempt_job)
        job["remain"] += self.preempt_overhead(job)

    def spot_runtime_log(self):
        failed, succeed = 0, 0
        for vc in self.cluster.vc_dict:
            spot_failed_ratio = 0
            spot_total_num = self.spot_failed[vc] + self.spot_succeed[vc]
            if spot_total_num > 0:
                spot_failed_ratio = round(self.spot_failed[vc] / spot_total_num, 4)
            self.logger.info(f"VC: {vc} | {spot_total_num} spot's guarantee hours | "
                             f"Preemption_rate {spot_failed_ratio} | "
                             f"{self.spot_succeed[vc]} succeed | {self.spot_failed[vc]} preempted")
            failed += self.spot_failed[vc]
            succeed += self.spot_succeed[vc]

        spot_failed_ratio = 0
        spot_total_num = failed + succeed
        if spot_total_num > 0:
            spot_failed_ratio = round(failed / spot_total_num, 4)
        self.logger.info(f"Cluster: {self.cluster_name} | {spot_total_num} spot's guarantee hours | "
                         f"Preemption_rate {spot_failed_ratio} | "
                         f"{succeed} succeed | {failed} preempted")

    def place_with_preemption(self, job):
        gpu_request = job["gpu_request"]
        vc = self.cluster.vc_dict[job["vc_name"]]

        if gpu_request < 1.0:
            flag, node, gpu, preempt_spots = self.frag_preempt(job)
            if flag:
                for spot in preempt_spots:
                    vc.release_resource(spot)
                assert node.allocate_share_gpu(gpu, job)
                job["nodes"].append({node.node_name: [gpu]})
                return True, preempt_spots

        flag, alloc_node, alloc_gpus, preempt_spots = self.select_with_preemption(job)
        if flag:
            for spot in preempt_spots:
                vc.release_resource(spot)
            assert alloc_node.allocate_gpus_with_preemption(alloc_gpus, job)
            job["nodes"].append({alloc_node.node_name: alloc_gpus})
            return True, preempt_spots
        return False, None

    # 碎卡抢占
    def frag_preempt(self, job):
        job_gpu_num = job["gpu_request"]
        vc = self.cluster.vc_dict[job["vc_name"]]

        # 筛选可行节点，模拟调度器Filter阶段
        frag_gpus = []
        min_gpu = 1.0
        for node in vc.node_list:
            for i in range(node.num_gpus):
                galloc = node.node_type_galloc[i]["HP"]
                if 0.0 < galloc < 1.0 - job_gpu_num:
                    if 1.0 - galloc < min_gpu:
                        frag_gpus = [[node, i, 1.0 - galloc]]
                        min_gpu = 1.0 - galloc
                    elif 1.0 - galloc == min_gpu:
                        frag_gpus.append([node, i, 1.0 - galloc])

        # 寻找最优节点，模拟调度器Score阶段
        best_node, best_gpu, best_spots = None, None, None
        if len(frag_gpus) == 0:
            return False, best_node, best_gpu, best_spots
        min_cost = 20.0
        for card in frag_gpus:
            node, gpu = card[0], card[1]
            cost, spots = self.frag_preempt_cost(job, node, gpu)
            if cost < min_cost:
                best_node, best_gpu, best_spots = node, gpu, spots
                min_cost = cost
        return True, best_node, best_gpu, best_spots

    # 碎卡抢占Score函数
    def frag_preempt_cost(self, guar_job, node, gpu):
        jobs = node.node_gpu_dict[gpu]
        free_frag_gpu = 1.0 - node.node_type_galloc[gpu]["HP"]
        preempt_cost = {}
        for job in jobs:
            if job["type"] == "Spot" and job["job_index"] not in preempt_cost:
                worker_num = 0  # 本卡上的副本数
                for node_dict in job["nodes"]:
                    for k, v in node_dict.items():
                        if k == node.node_name and gpu in v:
                            worker_num += 1
                assert worker_num > 0
                cost = job.get_preempt_cost(self.time)
                preempt_cost[job["job_index"]] = (job, cost, worker_num * job["gpu_request"])

        # 候选Spot集合为空
        if len(preempt_cost) == 0:
            return 0, set()

        preempt_cost = sorted(preempt_cost.values(), key=lambda x: x[1], reverse=True)
        preempt_spots = []
        l, r = 0, len(preempt_cost) - 1
        while l < r:
            while l <= r and free_frag_gpu - preempt_cost[l][2] >= guar_job["gpu_request"]:
                free_frag_gpu -= preempt_cost[l][2]
                l += 1
            if l > r:
                break
            preempt_spots.append(preempt_cost[l][0])
            l += 1
            while l <= r and free_frag_gpu - preempt_cost[r][2] >= guar_job["gpu_request"]:
                free_frag_gpu -= preempt_cost[r][2]
                r -= 1
            if l > r:
                break
            preempt_spots.append(preempt_cost[r][0])
            r -= 1
        cost = self.calc_preempt_cost(guar_job["vc_name"], preempt_spots)
        return cost, preempt_spots

    # 整卡抢占
    def select_with_preemption(self, job):
        job_gpu_num = int(max(job["gpu_request"], 1))
        vc = self.cluster.vc_dict[job["vc_name"]]

        # 筛选可行节点，模拟调度器Filter阶段
        avail_node_list = []
        min_gpu = 16
        for node in vc.node_list:
            free_gpus = node.check_free_gpus_with_preemption()
            if job_gpu_num <= free_gpus:
                if free_gpus < min_gpu:
                    avail_node_list = [node]
                    min_gpu = free_gpus
                elif free_gpus == min_gpu:
                    avail_node_list.append(node)

        # 寻找最优节点，模拟调度器Score阶段
        best_node, best_gpus, best_spots = None, None, None
        if len(avail_node_list) == 0:
            return False, best_node, best_gpus, best_spots
        min_cost = 20.0
        for node in avail_node_list:
            cost, gpus, spots = self.preempt_cost(job, node)
            if cost < min_cost:
                best_node, best_gpus, best_spots = node, gpus, spots
                min_cost = cost
        return True, best_node, best_gpus, best_spots

    # 整卡抢占Score函数
    def preempt_cost(self, guar_job, node):
        job_gpu_num = int(max(guar_job["gpu_request"], 1))
        candidate_gpus = set()
        preempt_cost = {}
        for k, v in node.node_type_galloc.items():
            if v["HP"] == 0.0:
                candidate_gpus.add(k)
                jobs = node.node_gpu_dict[k]
                for job in jobs:
                    if job["job_index"] not in preempt_cost:
                        job_gpus = set()
                        for node_dict in job["nodes"]:
                            for kk, vv in node_dict.items():
                                if kk == node.node_name:
                                    job_gpus.update(vv)
                        cost = job.get_preempt_cost(self.time)
                        preempt_cost[job["job_index"]] = (job, cost, job_gpus)
                    # self.logger.info(f"card: {k} | job: {job}")

        # self.logger.info(f"{node.node_name} | candidate_gpus: {candidate_gpus} "
        #                  f"| node_type_galloc: {node.node_type_galloc}")

        # 候选Spot集合为空
        if len(preempt_cost) == 0:
            return 0, list(candidate_gpus)[:job_gpu_num], set()

        preempt_cost = sorted(preempt_cost.values(), key=lambda x: x[1], reverse=True)
        preempt_spots = []
        l, r = 0, len(preempt_cost) - 1
        while l <= r:
            while l <= r and len(candidate_gpus - preempt_cost[l][2]) >= job_gpu_num:
                candidate_gpus = candidate_gpus - preempt_cost[l][2]
                l += 1
            if l > r:
                break
            preempt_spots.append(preempt_cost[l][0])
            l += 1
            while l <= r and len(candidate_gpus - preempt_cost[r][2]) >= job_gpu_num:
                candidate_gpus = candidate_gpus - preempt_cost[r][2]
                r -= 1
            if l > r:
                break
            preempt_spots.append(preempt_cost[r][0])
            r -= 1
        cost = self.calc_preempt_cost(guar_job["vc_name"], preempt_spots)
        return cost, list(candidate_gpus)[:job_gpu_num], preempt_spots

    # 抢占对优化目标的损失计算
    def calc_preempt_cost(self, vc_name, preempt_spots):
        running_spots_num = 0
        total_cost = 0.0
        for spot in preempt_spots:
            total_cost += spot.get_preempt_cost(self.time)
            if spot["status"] == "run":
                running_spots_num += 1
        vc = self.cluster.vc_dict[vc_name]
        cost = ((self.spot_failed[vc_name] + running_spots_num) /
                (self.spot_failed[vc_name] + self.spot_succeed[vc_name] + running_spots_num)) + (
                       5.0 * total_cost / (vc.total_gpus * self.timespan))
        return cost

    def record_spot_failed(self, vc_name, preempt_job):
        preempt_job["preempt_times"] += 1
        self.preemption_times += 1
        self.spot_failed[vc_name] += 1
        vc = self.cluster.vc_dict[vc_name]
        vc.record_spot(self.time, preempt_job, succeed=0)

    def record_spot_succeed(self, job):
        vc_name = job["vc_name"]
        self.spot_succeed[vc_name] += 1
        vc = self.cluster.vc_dict[vc_name]
        vc.record_spot(self.time, job, succeed=1)
        if self.args.log_range[0] < self.time <= self.args.log_range[1]:
            if job["remain"] > 0:
                self.spot_gpu_time[vc_name] += job["gpu_request"] * job["worker_num"] * self.args.ckpt_interval
            else:
                self.spot_gpu_time[vc_name] += (job["gpu_request"] * job["worker_num"] *
                                                (job["duration"] % self.args.ckpt_interval))
