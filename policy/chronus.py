import numpy as np

from .policy import Policy
import pandas as pd
from estimator.gpu_request_estimator import GPURequestEstimator
from scipy.stats import norm


class Chronus(Policy):
    def __init__(self, trace, cluster, args, log_dir, logger):
        super(Chronus, self).__init__(trace, cluster, args, log_dir, logger)
        self._name = "chronus"
        self.be_interval = 300
        self.guar_interval = 1200

        self.spot_failed, self.spot_succeed = self.init_count_dict(), self.init_count_dict()  # Spot的保障周期成功/失败率

    def simulate(self):
        prev_index = 0

        while self.end_job_num != self.total_job_num:
            prev_time = max(0, self.time - self.be_interval)

            """1. Check & Release End Jobs"""
            for vc_name in self.run_list:
                run_ls = self.run_list[vc_name].copy()  # Avoid list.remove() issue
                vc = self.cluster.vc_dict[vc_name]
                for job in run_ls:
                    run_time = min(job["remain"], self.be_interval)
                    job["remain"] -= run_time
                    if self.args.log_range[0] < self.time <= self.args.log_range[1]+1:
                        if job["type"] == "HP":
                            self.guar_gpu_time[vc_name] += job["gpu_request"] * job["worker_num"] * run_time
                        else:
                            self.spot_gpu_time[vc_name] += job["gpu_request"] * job["worker_num"] * run_time

                    if job["type"] == "Spot":
                        self.record_spot_succeed(job)

                    if job["remain"] == 0:
                        job["status"] = "end"
                        job["end_time"] = prev_time + run_time
                        self.end_job_num += 1
                        assert vc.release_resource(job)
                        self.run_list[vc_name].remove(job)
                        self.gpu_request_dict[vc_name][job["type"]] -= job["gpu_request"] * job["worker_num"]


            """2. Flush All Lease"""
            if self.time >= self.args.log_range[0] and self.time % self.guar_interval == 0:
                for vc_name in self.run_list:
                    run_ls = self.run_list[vc_name].copy()  # Avoid list.remove() issue
                    for job in run_ls:
                        vc = self.cluster.vc_dict[vc_name]
                        assert vc.release_resource(job)
                        self.preempt(vc_name, job)

            """3. Allocate New / Pending Jobs"""
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
            self.alloc_batch_jobs(self.guar_que_list)
            self.alloc_batch_jobs(self.spot_que_list)

            """4. Log & Result Recorder"""
            self.recorder.update_seq_recorder()

            if self.time % 86400 == 0:
                self.runtime_log()
                if self.args.log_range[0] <= self.time:
                    self.spot_runtime_log()

            self.time += self.be_interval

        self.recorder.log_recorder(self._name, self.args.log_range)
        # self.spot_scheduler_log(log_range)

    def pend_job(self, job):
        job["status"] = "pend"
        job.set_preempt_time(self.time)
        if job["type"] == "Spot":
            self.spot_que_list[job["vc_name"]].append(job)
        else:
            self.guar_que_list[job["vc_name"]].append(job)

    def alloc_batch_jobs(self, job_lists):
        for vc in job_lists:
            job_lists[vc].sort(key=lambda x: (-x.__getitem__("gpu_request"),
                                              x.__getitem__("worker_num"),
                                              x.__getitem__("submit_time")))
            que_ls = job_lists[vc].copy()  # Avoid list.remove() issue
            failed_gpu_num = 16

            for job in que_ls:
                if job["gpu_request"] >= failed_gpu_num:
                    continue
                if self.job_placer(job):
                    if job["type"] == "HP":
                        self.run_guar(job)
                    else:
                        self.run_spot(job)
                else:
                    failed_gpu_num = job["gpu_request"]

    def run_spot(self, job):
        job["start_time"] = self.time
        job["queue"] = job["queue"] + (self.time - job.get_preempt_time())
        job["status"] = "run"
        if job["type"] == "Spot":
            job.update({"spot_duration": 3600})
            self.spot_que_list[job["vc_name"]].remove(job)
        else:
            raise ValueError
        self.run_list[job["vc_name"]].append(job)
        self.gpu_request_dict[job["vc_name"]][job["type"]] += job["gpu_request"] * job["worker_num"]

    def preempt(self, vc_name, preempt_job):
        preempt_job.update({"nodes": []})
        preempt_job.set_preempt_time(self.time)
        if preempt_job["status"] == "run":
            preempt_job["remain"] += 120 # preemption overhead
            self.run_list[vc_name].remove(preempt_job)
            self.gpu_request_dict[vc_name][preempt_job["type"]] -= (
                    preempt_job["gpu_request"] * preempt_job["worker_num"])
            self.pend_job(preempt_job)

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

    def record_spot_succeed(self, job):
        vc_name = job["vc_name"]
        self.spot_succeed[vc_name] += 1
        vc = self.cluster.vc_dict[vc_name]
        vc.record_spot(self.time, job, succeed=1)
