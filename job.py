import sys


class Job(dict):
    def __init__(self, series):
        super(Job, self).__init__()
        self.update(series.to_dict())
        # Priority Define by Estimator, Random Means No History Data Found
        self.update({"nodes": [], "priority": -1, "random": 0})

    def get_preempt_cost(self, time):
        if self["status"] == "run":
            return self["worker_num"] * self["gpu_request"] * (time - self["start_time"])
        return 0

    def set_ckpt_time(self, time):
        self.last_ckpt_time = time

    def get_ckpt_time(self):
        return self.last_ckpt_time

    def set_preempt_time(self, time):
        self.last_preempt_time = time

    def get_preempt_time(self):
        return self.last_preempt_time


class Trace:
    def __init__(self):
        self.job_list = []

    def append_job(self, job):
        self.job_list.append(job)

    def job_num(self):
        return len(self.job_list)

    def sort_jobs(self, key):
        self.job_list.sort(key=lambda x: x.__getitem__(key))

    def cluster_trace(self):
        cluster_trace = Trace()
        for job in self.job_list:
           cluster_trace.append_job(job)
        cluster_trace.sort_jobs("submit_time")
        return cluster_trace