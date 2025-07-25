import numpy as np

from .policy import Policy
import pandas as pd
from estimator.gpu_request_estimator import GPURequestEstimator
from scipy.stats import norm


class FGD(Policy):
    def __init__(self, trace, cluster, args, log_dir, logger):
        super(FGD, self).__init__(trace, cluster, args, log_dir, logger)
        self._name = "FGD"
        self.timespan = self.args.log_range[1] - self.args.log_range[0] + 1
        self.estimator = GPURequestEstimator(args)
        self.spot_failed, self.spot_succeed = self.init_count_dict(), self.init_count_dict()  # Spot的保障周期成功/失败率
        self.typicalPods = {1: 0.35, 2: 0.1, 4: 0.05, 8: 0.5}

    def simulate(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:

            """0. Update Spot Quota"""
            if self.args.log_range[0] <= self.time <= self.args.log_range[1] and self.time % 300 == 0:
                now_timestamp = pd.Timestamp(self.args.trace_range[0]) + pd.Timedelta(seconds=self.time)
                self.logger.info(
                    f"{self.cluster_name} | Simulation Time: {now_timestamp}"
                )
                vc_record_df = self.recorder.agg_seq_recorder()
                if self.time % 86400 == 0:
                    self.estimator.train(vc_record_df, now_timestamp)
                self.update_spot_quota(vc_record_df, now_timestamp)
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
                        self.gpu_request_dict[vc_name][job["type"]] -= job["gpu_request"] * job["worker_num"]
                        if job["type"] == "Spot":
                            self.record_spot_succeed(job)
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
                self.guar_que_list[vc].sort(key=lambda x: x.__getitem__("submit_time"))
                que_ls = self.guar_que_list[vc].copy()  # Avoid list.remove() issue

                for job in que_ls:
                    place_flag, preempt_jobs = self.place_with_preemption(job)
                    if place_flag:
                        self.run_guar(job)
                        self.preempt(job, preempt_jobs)
                    else:
                        break

            for key in self.spot_que_list:
                vc = self.cluster.vc_dict[key]
                self.spot_que_list[key].sort(key=lambda x: x.__getitem__("submit_time"))
                spot_que_ls = self.spot_que_list[key].copy()

                # candidate_vcs = self.cluster.spot_vc[key]
                for job in spot_que_ls:
                    if ((self.time > self.args.log_range[1]) or
                            ((self.args.log_range[0] <= self.time <= self.args.log_range[1]) and
                            (self.gpu_request_dict[key]["Spot"] + job["gpu_request"] * job["worker_num"] <=
                             self.spot_quota[key]["quota"][job["guarantee_request"]]))):
                        if self.spot_place(job):
                            self.run_spot(job)
                        else:
                            vc.release_resource(job)
                            job.update({"nodes": []})
                            break

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
            # key = job["gpu_model"] + '&&' + job["cluster_name"]
            # self.spot_que_list[key].remove(job)
        else:
            raise ValueError
        self.run_list[job["vc_name"]].append(job)
        self.gpu_request_dict[job["vc_name"]][job["type"]] += job["gpu_request"] * job["worker_num"]

    def preempt(self, job, preempt_jobs):
        vc_name = job["vc_name"]
        for preempt_job in preempt_jobs:
            preempt_job.update({"nodes": []})
            if preempt_job["status"] == "run":
                preempt_job["remain"] = (preempt_job["remain"] + self.preempt_overhead(preempt_job) +
                                         (self.time - preempt_job["start_time"]) % self.args.ckpt_interval)

                self.run_list[vc_name].remove(preempt_job)
                self.gpu_request_dict[vc_name][preempt_job["type"]] -= (
                        preempt_job["gpu_request"] * preempt_job["worker_num"])

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
                if 0 < galloc < 1.0 - job_gpu_num:
                    if 1.0 - galloc < min_gpu:
                        frag_gpus = [[node, i, 1.0 - galloc]]
                        min_gpu = 1.0 - galloc

        # 寻找最优节点，模拟调度器Score阶段
        if len(frag_gpus) == 0:
            return False, None, None, None
        node, gpu = frag_gpus[0][0], frag_gpus[0][1]
        spots = self.frag_preempt_cost(job_gpu_num, node, gpu)
        return True, node, gpu, spots

    # 整卡抢占
    def select_with_preemption(self, job):
        job_gpu_num = int(max(job["gpu_request"], 1))
        vc = self.cluster.vc_dict[job["vc_name"]]

        # 筛选可行节点，模拟调度器Filter阶段
        avail_node_list = []
        min_frag_delta = 16.0
        for node in vc.node_list:
            free_gpus = node.check_free_gpus_with_preemption()
            if job_gpu_num <= free_gpus:
                frag_delta = self.frag_delta(free_gpus, job_gpu_num)
                if frag_delta < min_frag_delta:
                    avail_node_list = [node]
                    min_frag_delta = frag_delta

        # 寻找最优节点，模拟调度器Score阶段
        if len(avail_node_list) == 0:
            return False, None, None, None
        node = avail_node_list[0]
        gpus, spots = self.preempt_cost(job_gpu_num, node)
        return True, node, gpus, spots

    # 碎卡抢占Score函数
    @staticmethod
    def frag_preempt_cost(job_gpu_num, node, gpu):
        preempt_spots = []
        preempt_spots_index = set()
        free_frag_gpu = 1.0 - node.node_galloc[gpu]
        if free_frag_gpu < job_gpu_num:
            jobs = node.node_gpu_dict[gpu]
            for job in jobs:
                if job["type"] == "Spot":
                    if job["job_index"] not in preempt_spots_index:
                        preempt_spots_index.add(job["job_index"])
                        preempt_spots.append(job)
                    free_frag_gpu = free_frag_gpu + job["gpu_request"]
                    if free_frag_gpu >= job_gpu_num:
                        break
        return preempt_spots

    # 整卡抢占Score函数
    @staticmethod
    def preempt_cost(job_gpu_num, node):
        gpus = []
        preempt_spots = []
        preempt_spot_indexes = set()
        for k, v in node.node_gpu_dict.items():
            priority_galloc = node.node_type_galloc[k]["HP"]
            if priority_galloc == 0.0:
                gpus.append(k)
                for inst in v:
                    if inst["job_index"] not in preempt_spot_indexes:
                        preempt_spot_indexes.add(inst["job_index"])
                        preempt_spots.append(inst)
                if len(gpus) == job_gpu_num:
                    break
        return gpus, preempt_spots

    def spot_place(self, job):
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
        vc = self.cluster.vc_dict[job["vc_name"]]

        # 筛选可行节点，模拟调度器Filter阶段
        frag_gpus = []
        min_gpu = 1.0
        for node in vc.node_list:
            for i in range(node.num_gpus):
                galloc = node.node_galloc[i]
                if 0 < galloc < 1.0 - job_gpu_num:
                    if 1.0 - galloc < min_gpu:
                        frag_gpus = [[node, i, 1.0 - galloc]]

        # 寻找最优节点，模拟调度器Score阶段
        if len(frag_gpus) == 0:
            return False, None, None
        node, gpu = frag_gpus[0][0], frag_gpus[0][1]
        return True, node, gpu

    # 整卡抢占
    def gpu_select(self, job):
        job_gpu_num = int(max(job["gpu_request"], 1))
        vc = self.cluster.vc_dict[job["vc_name"]]

        # 筛选可行节点，模拟调度器Filter阶段
        avail_node_list = []
        min_frag_delta = 16
        for node in vc.node_list:
            free_gpus = node.free_gpus
            if job_gpu_num <= free_gpus:
                frag_delta = self.frag_delta(free_gpus, job_gpu_num)
                if frag_delta < min_frag_delta:
                    avail_node_list = [node]
                    min_frag_delta = frag_delta

        # 寻找最优节点，模拟调度器Score阶段
        if len(avail_node_list) == 0:
            return False, None
        node = avail_node_list[0]
        return True, node

    def frag_delta(self, free_gpu_num, job_gpu_num):
        old_frag = self.frag_func(free_gpu_num)
        new_frag = self.frag_func(free_gpu_num-job_gpu_num)
        return new_frag - old_frag

    def frag_func(self, free_gpu_num):
        delta = 0
        for pod, weight in self.typicalPods.items():
            delta += weight * (free_gpu_num - (free_gpu_num // pod))
        return delta

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
        if self.args.log_range[0] <= self.time <= self.args.log_range[1]:
            if job["remain"] > 0:
                self.spot_gpu_time[vc_name] += job["gpu_request"] * job["worker_num"] * self.args.ckpt_interval
            else:
                self.spot_gpu_time[vc_name] += job["gpu_request"] * job["worker_num"] * (job["duration"] % self.args.ckpt_interval)
