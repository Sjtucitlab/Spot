import numpy as np
import pandas as pd


def read_cluster_trace(cluster_trace):
    cluster_df = pd.read_csv(cluster_trace)
    cluster_df["vc_name"] = cluster_df["gpu_model"]
    return cluster_df


class Cluster:
    def __init__(self, cluster_trace, cluster_name="cluster"):
        self.cluster_name = cluster_name
        self.cluster_df = read_cluster_trace(cluster_trace)
        self.vc_num = self.cluster_df['vc_name'].nunique()
        self.node_num = self.cluster_df.shape[0]
        self.vc_list = []
        self.vc_dict = {}
        self.total_gpus = 0
        self.total_cpus = 0
        self.init_cluster_vc()

    def init_cluster_vc(self):
        for vc_name in self.cluster_df["vc_name"].unique():
            vc_info_df = self.cluster_df[self.cluster_df["vc_name"] == vc_name]
            vc = VC(vc_name, vc_info_df)
            self.vc_list.append(vc_name)
            self.vc_dict[vc_name] = vc
            self.total_cpus += vc.total_cpus
            self.total_gpus += vc.total_gpus


# Virtual Cluster
class VC:
    def __init__(self, vc_name, vc_info_df):
        self.vc_name = vc_name
        self.node_num = vc_info_df.shape[0]

        self._num_gpus_per_node = vc_info_df.gpu_capacity_num.max()
        self._num_cpus_per_node = vc_info_df.cpu_num.max()
        self.total_gpus = 0
        self.total_cpus = 0

        self.node_list = []
        self.node_dict = {}
        self.init_vc_node(vc_info_df)

        self.colocate_enable = 0
        # Temp Node num with additional temp_node_num_base
        self.temp_node_num_base = 9999
        self.has_temp_node = False  # To avoid first-time scaling error

        self.spot_log = []
        self.spot_log_df = pd.DataFrame(columns=['time', 'succeed', 'node', 'count'])
        self.eviction_score = self.init_eviction_score()
        self.hour_weight, self.penalty_power = 0.8, 2

    def init_vc_node(self, vc_info: pd.DataFrame):
        for idx, row in vc_info.iterrows():
            new_node = Node(node_name=row['node_name'], num_gpus=int(row['gpu_capacity_num']),
                            num_cpus=int(row['cpu_num']))
            self.node_list.append(new_node)
            self.node_dict[row['node_name']] = new_node
            self.total_gpus = self.total_gpus + int(row['gpu_capacity_num'])
            self.total_cpus = self.total_cpus + int(row['cpu_num'])

    def check_node_inside_vc(self, node_id):
        return node_id in self.node_dict

    def check_node_inside_idle_vc(self, node_id):
        idle_list = self.idle_node_list()
        for i in idle_list:
            if i.node_name == node_id:
                return True
        return False

    def add_new_node(self, change_node_num, force_same_node):
        for i in range(change_node_num):
            temp_node_num = i + self.temp_node_num_base
            if self.check_node_inside_vc(temp_node_num) and force_same_node:
                # temp_node_num = temp_node_num + 1000
                # raise ValueError("Temp node num already exists")
                return False
            node = Node(temp_node_num, self._num_gpus_per_node, self._num_gpus_per_node)
            self.node_list.append(node)
        self.node_num = self.node_num + change_node_num
        self.total_gpus = self._num_gpus_per_node * self.node_num
        self.total_cpus = self._num_cpus_per_node * self.node_num

    def exchange_node_status(self, idle_node, i):
        # Just for simple simulation implementation in some rare cases.
        # In reality, we can directly remove different nodes.
        assert idle_node.check_free_gpus() == self._num_gpus_per_node
        temp_node = self.get_node(self.temp_node_num_base + i)
        temp_node.update_node_name(idle_node.node_name)
        idle_node.update_node_name(self.temp_node_num_base + i)
        temp_node.exchange_job_status()

    def remove_idle_node(self, change_node_num, force_same_node):
        idle_node_list = self.idle_node_list()
        if len(idle_node_list) < abs(change_node_num):
            return False  # Not enough idle nodes
        idle_node_list.sort(key=lambda x: x.node_name, reverse=True)
        idle_node_list = idle_node_list[: abs(change_node_num)]
        for i in range(abs(change_node_num)):
            if idle_node_list[i].node_name < self.temp_node_num_base and force_same_node and self.has_temp_node:
                self.exchange_node_status(idle_node_list[i], i)
                idle_node_list = self.idle_node_list()
                idle_node_list.sort(key=lambda x: x.node_name, reverse=True)
                assert idle_node_list[0].node_name >= self.temp_node_num_base
            to_remove_node = idle_node_list[i]
            self.node_list.remove(to_remove_node)
        self.has_temp_node = True
        assert len(self.node_list) == self.node_num + change_node_num
        self.node_num = self.node_num + change_node_num
        self.total_gpus = self._num_gpus_per_node * self.node_num
        self.total_cpus = self._num_cpus_per_node * self.node_num
        return True

    def update_vc_node(self, change_node_num, force_same_node=True):
        if change_node_num > 0:
            self.add_new_node(change_node_num, force_same_node)
        elif change_node_num < 0:
            self.remove_idle_node(change_node_num, force_same_node)
        else:
            raise ValueError("`change_node_num` should not be 0")

    def get_node(self, node_id):
        if node_id not in self.node_dict:
            return None
        return self.node_dict[node_id]

    def vc_free_gpus(self):
        return sum(node.free_gpus for node in self.node_list)

    def vc_free_cpus(self):
        return sum(node.free_cpus for node in self.node_list)

    def vc_gpu_request(self):
        vc_gpu_request = {}
        for node in self.node_list:
            gpu_request_dict = node.check_gpu_request()
            for type, value in gpu_request_dict.items():
                if type in vc_gpu_request:
                    vc_gpu_request[type] += value
                else:
                    vc_gpu_request[type] = value
        return vc_gpu_request

    def idle_node_list(self):
        idle_node_list = []
        for node in self.node_list:
            if node.free_gpus == self._num_gpus_per_node:
                idle_node_list.append(node)
        return idle_node_list

    def avail_node_list(self, job_gpu_num=1):
        avail_node_list = []
        for node in self.node_list:
            if node.free_gpus >= job_gpu_num:
                avail_node_list.append(node)
        return avail_node_list

    def scheduler_score_dict(self, job_gpu_num, job_type):
        scheduler_score_dict = {}
        for node in self.node_list:
            score = node.get_scheduler_score(job_gpu_num, job_type)
            if score >= 0:
                scheduler_score_dict[node.node_name] = score
        return scheduler_score_dict

    def frag_gpu_dict(self, job_gpu_num=0):
        frag_gpu_dict = {}
        for node in self.node_list:
            for i in range(node.num_gpus):
                if 0.0 < node.node_galloc[i] <= 1.0 - job_gpu_num:
                    frag_gpu_dict[(node, i)] = 1.0 - node.node_galloc[i]
        return dict(sorted(frag_gpu_dict.items(), key=lambda x: x[1]))

    def avail_node_list_with_preemption(self, job_gpu_num=1):
        avail_node_list = []
        for node in self.node_list:
            if node.check_free_gpus_with_preemption() >= job_gpu_num:
                avail_node_list.append(node)
        return avail_node_list

    def release_resource(self, job):
        nodes_list = job["nodes"]
        for d in nodes_list:
            for i, gpu_list in d.items():
                node = self.get_node(i)
                assert node.node_name == i
                assert node.release_gpu(gpu_list, job)
        return True

    def record_spot(self, time, job, succeed=1):
        nodes_list = job["nodes"]
        for dict in nodes_list:
            for i, gpu_list in dict.items():
                self.spot_log.append({"time": time, "succeed": succeed, "node": i, "count": 1})

    def check_vc_colocate_jobs(self, job):
        # nodes_list = job["nodes"]
        # recover_jobs = set()
        # for dict in nodes_list:
        #     for i, gpu_list in dict.items():
        #         node = self.node_list[i]
        #         jobs = node.check_colocate_jobs(gpu_list, job)
        #         recover_jobs |= set(jobs)
        # return list(recover_jobs)
        nodes_list = job["nodes"]
        dict = nodes_list[0]
        for i, gpu_list in dict.items():
            node = self.get_node(i)
            colo_job_id = node.check_colocate_jobs(gpu_list, job)
            if colo_job_id:
                return colo_job_id
            else:
                raise NotImplementedError

    # Only one job running in a node
    def consolidate_node_num(self):
        list = []
        for node in self.node_list:
            if node.job_num == 1:
                list.append(node)
        return len(list)

    def shared_node_num(self):
        list = []
        for node in self.node_list:
            if node.job_num > 1:
                list.append(node)
        return len(list)

    def init_eviction_score(self):
        spot_score_dict = {}
        for node in self.node_list:
            spot_score_dict[node.node_name] = 10000
        return spot_score_dict

    def get_eviction_rate(self, time, hours):
        spot_eviction = pd.DataFrame(self.spot_log, columns=['time', 'node', 'count'])
        self.spot_log_df = pd.concat([self.spot_log_df, spot_eviction])
        self.spot_log_df = self.spot_log_df[self.spot_log_df.time >= time - 3600 * 24]
        self.spot_log = []

        recent_log_df = self.spot_log_df[self.spot_log_df.time >= time - 3600 * hours]
        succeed, failed = 0, 0
        if recent_log_df[recent_log_df.succeed == 0].shape[0] > 0:
            failed = recent_log_df[recent_log_df.succeed == 0]["count"].sum()
        if recent_log_df[recent_log_df.succeed == 1].shape[0] > 0:
            succeed = recent_log_df[recent_log_df.succeed == 1]["count"].sum()
        return succeed, failed

    def update_eviction_score(self, time):
        spot_score = []
        for node in self.node_list:
            spot_score.append([node.node_name, 10000])
        spot_score_df = pd.DataFrame(spot_score, columns=['node', 'spot_score'])

        spot_eviction = self.spot_log_df[self.spot_log_df.succeed == 0].groupby(['time', 'node']).agg({"count": "sum"})

        if spot_eviction.shape[0] > 0:
            spot_eviction_1h = spot_eviction[spot_eviction.time >= time - 3600]
            spot_eviction_1h = spot_eviction_1h.groupby(['node']).agg({"count": "sum"})
            spot_eviction_1h = spot_eviction_1h.rename(columns={"count": "count_1h"})
            spot_score_df = pd.merge(spot_score_df, spot_eviction_1h, how='left', on=['node']).fillna(0.0)

            spot_eviction_1d = spot_eviction[spot_eviction.time >= time - 3600 * 24]
            spot_eviction_1d = spot_eviction_1d.groupby(['node']).agg({"count": "sum"})
            spot_eviction_1d = spot_eviction_1d.rename(columns={"count": "count_1d"})
            spot_score_df = pd.merge(spot_score_df, spot_eviction_1d, how='left', on=['node']).fillna(0.0)

            spot_score_df['penalty_coef'] = (self.hour_weight * spot_score_df['count_1h'] +
                                             (1.0 - self.hour_weight) * spot_score_df['count_1d'])
            spot_score_df['penalty_coef'] = 1.0 - 0.01 * self.penalty_power ** spot_score_df['penalty_coef']
            spot_score_df.loc[spot_score_df.penalty_coef < 0.0, 'penalty_coef'] = 0.0
            spot_score_df['spot_score'] = spot_score_df['spot_score'] * spot_score_df['penalty_coef']

        self.eviction_score = spot_score_df.set_index('node')['spot_score'].to_dict()


class Node:
    def __init__(self, node_name, num_gpus, num_cpus):
        self.node_name = node_name
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.previous_node_name = node_name

        self.job_num = 0
        self.free_cpus = num_cpus
        # self.free_gpus = num_gpus
        # self.colocate_gpu_num = 0

        # self.node_job_dict = {}
        self.node_gpu_dict = self.init_gpu_dict()  # 分配的job alloc集合
        self.node_galloc = self.init_gpu_state()  # 已分配的GPU卡数
        self.node_type_galloc = self.init_gpu_state_type()  # 已分配的各类GPU卡数

    @property
    def free_gpus(self):
        return self.check_free_gpus()

    def init_gpu_dict(self):
        gdict = {}
        for i in range(self.num_gpus):
            gdict.update({i: []})
        return gdict

    def init_gpu_state(self):
        gdict = {}
        for i in range(self.num_gpus):
            gdict.update({i: 0})
        return gdict

    def init_gpu_state_type(self):
        gdict = {}
        for i in range(self.num_gpus):
            gdict.update({i: {"HP": 0, "Spot": 0}})
        return gdict

    def check_free_gpus(self):
        count = 0
        for k, v in self.node_gpu_dict.items():
            if len(v) == 0:
                count += 1
        return count

    def get_scheduler_score(self, job_gpu_num, job_type):
        free = 0
        exist = {"HP": 0, "Spot": 0}
        for k, v in self.node_type_galloc.items():
            if v["HP"] + v["Spot"] == 0:
                free += 1
            elif v["HP"] > 0:
                exist["HP"] += 1
            else:
                exist["Spot"] += 1
        if free < job_gpu_num:
            return -1
        score = (free + 1) * 100 - exist[job_type]
        return score

    def check_free_gpus_with_preemption(self):
        count = 0
        for k, v in self.node_type_galloc.items():
            if v["HP"] == 0.0:
                count += 1
        return count

    def check_gpu_request(self):
        gpu_request_dict = {}
        for k, v in self.node_gpu_dict.items():
            for job in v:
                if job["type"] not in gpu_request_dict.keys():
                    gpu_request_dict.update({job["type"]: job["gpu_request"]})
                else:
                    gpu_request_dict[job["type"]] += job["gpu_request"]
        return gpu_request_dict

    def check_free_gpu_list(self):
        free_list = []
        for k, v in self.node_gpu_dict.items():
            if len(v) == 0:
                free_list.append(k)
        return free_list

    def check_colocate_gpu_list(self):
        co_gpus = []
        for k, v in self.node_gpu_dict.items():
            if len(v) > 1:
                co_gpus.append(k)
        return co_gpus

    """allocate share gpu"""

    def allocate_share_gpu(self, gpu, job):
        self.job_num += 1

        self.node_gpu_dict[gpu].append(job)
        self.node_galloc[gpu] = self.node_galloc[gpu] + job["gpu_request"]
        assert self.node_galloc[gpu] <= 1.0

        job_type = job["type"]
        if job_type in self.node_type_galloc[gpu]:
            self.node_type_galloc[gpu][job_type] += job["gpu_request"]

        # self.node_job_dict.update({job["job_index"]: [gpu]})
        return True

    """allocate"""

    def allocate_gpu(self, num_gpu, job):
        assert num_gpu <= self.free_gpus
        # self.free_gpus -= num_gpu
        self.job_num += 1

        job_type = job["type"]

        allocate_gpus = []
        toallocate = num_gpu
        for k, v in self.node_gpu_dict.items():
            if toallocate == 0:
                break
            if not v:
                allocate_gpus.append(k)
                self.node_gpu_dict[k].append(job)
                self.node_galloc[k] = min(job["gpu_request"], 1.0)
                if job_type in self.node_type_galloc[k]:
                    self.node_type_galloc[k][job_type] += min(job["gpu_request"], 1.0)
                toallocate -= 1
        assert num_gpu == len(allocate_gpus)
        # self.node_job_dict.update({job["job_index"]: allocate_gpus})
        return allocate_gpus

    def allocate_gpus_with_preemption(self, gpus, job):
        self.job_num += 1
        job_type = job["type"]
        job_gpu_num = int(max(job["gpu_request"], 1))

        for gpu in gpus:
            self.node_gpu_dict[gpu].append(job)
            self.node_galloc[gpu] = min(job["gpu_request"], 1.0)

            if job_type in self.node_type_galloc[gpu]:
                self.node_type_galloc[gpu][job_type] += min(job["gpu_request"], 1.0)

        assert job_gpu_num == len(gpus)
        return True

    def select_preempt_spots(self, gpu, job_gpu_num, time):
        jobs = self.node_gpu_dict[gpu]
        free_frag_gpu = 1.0 - self.node_galloc[gpu]
        preempt_cost = []
        for job in jobs:
            if job["type"] == "Spot":
                preempt_cost.append((job, job.get_preempt_cost(time)))
        preempt_spots = set()
        while free_frag_gpu < job_gpu_num:
            for job in jobs:
                if job["type"] == "Spot":
                    preempt_spots.add(job)
                    free_frag_gpu = free_frag_gpu + job["gpu_request"]
        return preempt_spots

    """release"""

    def release_gpu(self, gpu_list, job):
        self.job_num -= 1
        job_type = job["type"]

        for i in gpu_list:
            assert isinstance(i, int)
            if job not in self.node_gpu_dict[i]:
                print("Job", job["job_index"], "not in node_gpu_dict")
            self.node_gpu_dict[i].remove(job)
            self.node_galloc[i] = self.node_galloc[i] - min(job["gpu_request"], 1.0)
            if job_type in self.node_type_galloc[i]:
                self.node_type_galloc[i][job_type] -= min(job["gpu_request"], 1.0)

        # self.node_job_dict.pop(job["job_index"])

        return True

    def update_node_name(self, new_name):
        # Echo `exchange_node_status`
        self.previous_node_name = self.node_name
        self.node_name = new_name

    def exchange_job_status(self):
        # Echo `exchange_node_status`
        jobs = []
        for k, v in self.node_gpu_dict.items():
            if v != []:
                for job in v:
                    if job not in jobs:
                        jobs.append(job)
        for job in jobs:
            for allocate_dict in job["nodes"]:
                k, v = list(allocate_dict.items())[0]
                if k == self.previous_node_name:
                    new_dict = {self.node_name: v}
                    job["nodes"].remove(allocate_dict)
                    job["nodes"].append(new_dict)

    # Future Extension
    def allocate_cpu(self, num_cpu):
        if num_cpu > self.free_cpus:
            return False
        else:
            self.free_cpus -= num_cpu
            return True

    def release_cpu(self, num_cpu):
        assert self.free_cpus + num_cpu <= self.num_cpus
        self.free_cpus += num_cpu
        return True


if __name__ == "__main__":
    trace_dir = "./data/Alibaba/node_info_df.csv"
    cluster = Cluster(trace_dir)

    print(cluster.vc_list)
