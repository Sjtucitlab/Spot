import numpy as np
from scipy.stats import norm

from .placer.consolidate import ConsolidatePlacement
from .placer.consolidateSpot import ConsolidateSpotPlacement

from .recorder.recorder import Recorder

class Policy:
    def __init__(self, trace, cluster, args, log_dir, logger):
        self._name = ""
        self._vc = ""
        self.args = args
        self.estimator = None
        self.timespan = self.args.log_range[1] - self.args.log_range[0] + 1

        self.cluster = cluster
        self.cluster_name = cluster.cluster_name
        self.trace = trace.cluster_trace()
        self.logger = logger

        self.spot_quota = self.init_spot_quota_dict()
        self.total_job_num = self.trace.job_num()
        self.spot_que_list = self.init_job_list_dict()

        self.guar_que_list = self.init_job_list_dict()
        self.gpu_request_dict = self.init_gpu_request_dict()
        self.guar_request_dict = {}
        self.spot_quota_record = []
        self.run_list = self.init_job_list_dict()  # Running Jobs
        self.guar_gpu_time, self.spot_gpu_time = self.init_count_dict(), self.init_count_dict()
        self.preemption_times = 0
        self.end_job_num = 0
        self.time = 0

        self.recorder = Recorder(self, log_dir)

        self.placer = self.init_placer()
        self.colo_df = None
        self.time_df = None

        self.vc_echo_scaling = False

    def init_placer(self):
        vc_placer = {}
        for vc in self.cluster.vc_dict:
            if self.args.scheduler == "spot_scheduler":
                vc_placer[vc] = ConsolidateSpotPlacement(self.cluster.vc_dict[vc])
            else:
                vc_placer[vc] = ConsolidatePlacement(self.cluster.vc_dict[vc])
        return vc_placer

    def init_gpu_request_dict(self):
        gpu_request_dict = {}
        for vc in self.cluster.vc_dict:
            gpu_request_dict[vc] = {"HP": 0, "Spot": 0}
        return gpu_request_dict

    def init_job_list_dict(self):
        job_list_dict = {}
        for vc in self.cluster.vc_dict:
            job_list_dict[vc] = []
        return job_list_dict

    def init_count_dict(self):
        count_dict = {}
        for vc in self.cluster.vc_dict:
            count_dict[vc] = 0
        return count_dict

    def update_guar_gpu_time(self):
        for vc in self.cluster.vc_dict:
            self.guar_gpu_time[vc] += self.gpu_request_dict[vc]["HP"]

    def job_placer(self, job):
        return self.placer[job["vc_name"]].place(job)

    def job_placer_with_preemption(self, job):
        return self.placer[job["vc_name"]].place_with_preemption(job)

    def update_spot_quota(self, org_request_df, now_timestamp):
        pred_mu, pred_sigma = self.estimator.test(org_request_df, now_timestamp)
        pred_mu_dict, pred_sigma_dict = {}, {}
        for i in range(len(org_request_df.columns)):
            org_name = org_request_df.columns[i]
            vc_name = org_name.split('&&', 1)[0]
            org_mu, org_sigma = pred_mu[0, :, i], pred_sigma[0, :, i]
            if vc_name not in pred_mu_dict:
                pred_mu_dict[vc_name] = org_mu
                pred_sigma_dict[vc_name] = org_sigma * org_sigma
            else:
                pred_mu_dict[vc_name] += org_mu
                pred_sigma_dict[vc_name] += org_sigma * org_sigma

        for vc_name in pred_mu_dict:
            vc = self.cluster.vc_dict[vc_name]
            gpu_capacity = vc.total_gpus
            vc_mu, vc_sigma = pred_mu_dict[vc_name], np.sqrt(pred_sigma_dict[vc_name])
            usage = norm.ppf(self.args.guarantee_rate, loc=vc_mu, scale=vc_sigma)
            free = gpu_capacity - np.clip(np.floor(usage), 0, gpu_capacity)

            # 调整额度安全系数eta
            succeed, faied = vc.get_eviction_rate(self.time, self.args.pred_len)
            if succeed + faied > 0:
                eviction_rate = float(faied) / float(succeed + faied)
                if eviction_rate >= 1.5 * (1.0 - self.args.guarantee_rate):
                    self.spot_quota[vc_name]["eta"] = (self.spot_quota[vc_name]["eta"] *
                                                       (1.0 - self.args.guarantee_rate) / eviction_rate)
                if (eviction_rate <= 0.5 * (1.0 - self.args.guarantee_rate) and
                        len(self.spot_que_list[vc_name]) > 0 and
                        self.time - self.spot_que_list[vc_name][0].get_preempt_time() > 3600):
                    self.spot_quota[vc_name]["eta"] = (self.spot_quota[vc_name]["eta"] *
                                                       (1.5 - (1.0 - self.args.guarantee_rate) / eviction_rate))

            # 更新spot额度
            new_spot_quota = {}
            for h in self.args.guarantee_hour:
                new_spot_quota[h] = np.min(free[:h]) * self.spot_quota[vc_name]["eta"]
                self.spot_quota_record.append([now_timestamp, vc_name, h, np.min(free[:h]),
                                               self.spot_quota[vc_name]["eta"]])
            self.spot_quota[vc_name]["quota"] = new_spot_quota

    def run_guar(self, job):
        job["start_time"] = self.time
        job["end_time"] = job["start_time"] + job["duration"]
        job["queue"] = self.time - job["submit_time"]
        job["status"] = "run"
        self.guar_que_list[job["vc_name"]].remove(job)
        self.run_list[job["vc_name"]].append(job)
        self.gpu_request_dict[job["vc_name"]][job["type"]] += job["gpu_request"]
        if job["org_key"] not in self.guar_request_dict:
            self.guar_request_dict[job["org_key"]] = job["gpu_request"]
        else:
            self.guar_request_dict[job["org_key"]] += job["gpu_request"]

    def init_spot_quota_dict(self):
        spot_quota_dict = {}
        for vc in self.cluster.vc_list:
            quota_dict = {}
            for i in self.args.guarantee_hour:
                quota_dict[i] = 0
            spot_quota_dict[vc] = {"quota": quota_dict, "eta": 1}
        return spot_quota_dict

    @staticmethod
    def preempt_overhead(job):
        gpu_num = job.__getitem__("gpu_request")
        if gpu_num <= 8:
            return 40
        else:
            return 60

    def runtime_log(self):
        for vc in self.run_list:
            self.logger.info(
                f"VC: {vc} | Time: {int(self.time)} | Running job: {len(self.run_list[vc])} "
                f"| Pending Guarantee job: {len(self.guar_que_list[vc])} "
                f"| Pending Spot job: {len(self.spot_que_list[vc])}"
            )
        guar_que_job_num = sum(len(self.guar_que_list[vc]) for vc in self.guar_que_list)
        spot_que_job_num = sum(len(self.spot_que_list[vc]) for vc in self.spot_que_list)
        run_job_num = sum(len(self.run_list[vc]) for vc in self.run_list)
        self.logger.info(
            f"{self.cluster_name} | Time: {int(self.time)} | Total Job: {self.total_job_num} "
            f"| End job: {self.end_job_num} | Running job: {run_job_num} "
            f"| Pending Guarantee job: {guar_que_job_num} | Pending Spot job: {spot_que_job_num} "
            f"| Preemption times: {self.preemption_times}"
        )
