import pandas as pd
import os


class Recorder:
    def __init__(self, policy, log_dir):
        self.policy = policy
        self._log_dir = log_dir
        self.logger = policy.logger

        # 单集群数据
        self.vc_record_columns = ["time", "vc_name", "total_gpu_num", "idle_gpu_num", "guar_gpu_request",
                                  "spot_gpu_request", "guar_que_gpu_num", "spot_que_gpu_num",
                                  "run_job_num",  "guar_que_job_num",  "spot_que_job_num",
                                  "total_node_num", "consolidate_node_num", "shared_node_num"]

        # 全局数据
        self.record_columns = ["time", "total_gpu_num", "idle_gpu_num", "guar_gpu_request", "spot_gpu_request",
                               "guar_que_gpu_num", "spot_que_gpu_num", "run_job_num", "guar_que_job_num",
                               "spot_que_job_num", "total_node_num", "consolidate_node_num", "shared_node_num"]

        # 成本组织维度GPU需求数据（OrgLinear算法用）
        self.org_request_columns = ["time", "org_key", "gpu_request"]

        # Spot额度数据
        self.spot_quota_columns = ["time", "vc_name", "guarantee_hour", "predict_free", "eta"]

        self.vc_record_df_list = []
        self.record_df_list = []
        self.org_request_df_list = []

        self.train_start_timestamp = pd.Timestamp("2024-03-01 20:00:00")

    def update_seq_recorder(self):
        vc_record_data = []
        for vc_name in self.policy.cluster.vc_dict:
            vc = self.policy.cluster.vc_dict[vc_name]
            vc_record_data.append([self.policy.time, vc_name, vc.total_gpus, vc.vc_free_gpus(),
                                   self.policy.gpu_request_dict[vc_name]["HP"],
                                   self.policy.gpu_request_dict[vc_name]["Spot"],
                                   sum(job.__getitem__("gpu_request") for job in self.policy.guar_que_list[vc_name]),
                                   sum(job.__getitem__("gpu_request") for job in self.policy.spot_que_list[vc_name]),
                                   len(self.policy.run_list[vc_name]), len(self.policy.guar_que_list[vc_name]),
                                   len(self.policy.spot_que_list[vc_name]), vc.node_num,
                                   vc.consolidate_node_num(), vc.shared_node_num()])

        update_vc_record_df = pd.DataFrame(columns=self.vc_record_columns, data=vc_record_data)
        update_record_df = update_vc_record_df.groupby(["time"]).agg(
            {"total_gpu_num": "sum", "idle_gpu_num": "sum", "guar_gpu_request": "sum",
             "spot_gpu_request": "sum", "guar_que_gpu_num": "sum", "spot_que_gpu_num": "sum",
             "run_job_num": "sum", "guar_que_job_num": "sum", "spot_que_job_num": "sum",
             "total_node_num": "sum", "consolidate_node_num": "sum", "shared_node_num": "sum"}).reset_index()

        self.vc_record_df_list.append(update_vc_record_df)
        self.record_df_list.append(update_record_df)

        org_request_data = []
        for org_key in self.policy.guar_request_dict.keys():
            org_request_data.append([self.policy.time, org_key, self.policy.guar_request_dict[org_key]])
        update_org_request_df = pd.DataFrame(columns=self.org_request_columns, data=org_request_data)
        self.org_request_df_list.append(update_org_request_df)

    def agg_seq_recorder(self):
        vc_record_df = pd.concat(self.vc_record_df_list)
        record_df = pd.concat(self.record_df_list)
        org_request_df = pd.concat(self.org_request_df_list)
        self.vc_record_df_list = [vc_record_df]
        self.record_df_list = [record_df]
        self.org_request_df_list = [org_request_df]
        return self.process_seq_data(org_request_df)

    def process_seq_data(self, org_request_df):
        data = org_request_df.copy()
        data['gpu_request'] = data['gpu_request'].astype(float)
        data['time'] = data['time'].astype(int)

        def transform_time(delta_seconds):
            return self.train_start_timestamp + pd.Timedelta(seconds=delta_seconds)

        data['time'] = data['time'].apply(transform_time)
        org_data = pd.DataFrame(index=data['time'].unique())
        for org in data.org_key.unique():
            org_data[org] = pd.Series(data=data[data.org_key == org].gpu_request.to_list(),
                                      index=data[data.org_key == org]['time'])
        org_data.fillna(0.0, inplace=True)
        return org_data

    def log_recorder(self, policy_name, log_range, spot_quota_record=None):
        vc_record_df = pd.concat(self.vc_record_df_list)
        record_df = pd.concat(self.record_df_list)
        # placement = self.policy.placement
        cluster_name = self.policy.cluster_name
        if not os.path.exists(os.path.join(self._log_dir, cluster_name)):
            os.mkdir(os.path.join(self._log_dir, cluster_name))

        df = pd.DataFrame(self.policy.trace.job_list)

        if len(df) == 0:
            print("No Job in cluster: ", cluster_name)
            raise NotImplementedError

        df["jct"] = df["end_time"] - df["submit_time"]

        df.to_csv(f"{self._log_dir}/{cluster_name}/{policy_name}_log.csv", index=False)

        record_df["gpu_utilization"] = ((record_df["total_gpu_num"] - record_df["idle_gpu_num"]) /
                                        record_df["total_gpu_num"]).round(3)
        record_df.to_csv(f"{self._log_dir}/{cluster_name}/{policy_name}_seq.csv", index=False)
        vc_record_df["gpu_utilization"] = ((vc_record_df["total_gpu_num"] - vc_record_df["idle_gpu_num"]) /
                                           vc_record_df["total_gpu_num"]).round(3)
        vc_record_df.to_csv(f"{self._log_dir}/{cluster_name}/{policy_name}_vc_record.csv", index=False)

        self.logger.info(f"{cluster_name}  | Log Time Start {log_range[0]} | Log Time End {log_range[1]}")
        log_df = df[(df.submit_time >= log_range[0]) & (df.submit_time <= log_range[1])]
        log_seq = record_df[(record_df.time >= log_range[0]) & (record_df.time <= log_range[1])]
        vc_seq = vc_record_df[(vc_record_df.time >= log_range[0]) & (vc_record_df.time <= log_range[1])]
        self.log_output(log_df, log_seq, vc_seq)

        if spot_quota_record is not None and len(spot_quota_record) > 0:
            spot_quota_df = pd.DataFrame(columns=self.spot_quota_columns, data=spot_quota_record)
            spot_quota_df["spot_quota"] = spot_quota_df["predict_free"] + spot_quota_df["eta"]

            spot_quota_df.to_csv(f"{self._log_dir}/{cluster_name}/{policy_name}_spot_quota.csv", index=False)

    def log_output(self, df, seq, vc_seq_record):
        cluster_name = self.policy.cluster_name
        for type in ["HP", "Spot"]:
            tmp_df = df[df["type"] == type]
            avg_jct = round(tmp_df["jct"].mean(), 2)
            avg_que = round(tmp_df["queue"].mean(), 2)
            avg_preemption_times = round(tmp_df["preempt_times"].mean(), 4)
            self.logger.info(f"{cluster_name} | {tmp_df.shape[0]} {type} jobs | Average JCT: {avg_jct} "
                             f"| Average Queue: {avg_que} | Average Preemption_rate: {avg_preemption_times}")

        avg_jct = round(df["jct"].mean(), 2)
        avg_que = round(df["queue"].mean(), 2)
        avg_preemption_times = round(df["preempt_times"].mean(), 4)
        avg_alloc = round(seq["gpu_utilization"].mean(), 4)
        avg_useful = ((sum(self.policy.guar_gpu_time.values()) + sum(self.policy.spot_gpu_time.values())) /
                      (self.policy.timespan * seq["total_gpu_num"].max()))

        self.logger.info(f"{cluster_name} | Average JCT: {avg_jct} | Average Queue: {avg_que} | Average "
                         f"Preemption_rate: {avg_preemption_times} | Average GPU Allocation: {avg_alloc} "
                         f"| Average Useful GPU-Time Rate: {avg_useful}")

        for vc_name in self.policy.cluster.vc_dict:
            for type in ["HP", "Spot"]:
                tmp_df = df[(df["vc_name"] == vc_name) & (df["type"] == type)]
                avg_jct = round(tmp_df["jct"].mean(), 2)
                avg_que = round(tmp_df["queue"].mean(), 2)
                avg_preemption_times = round(tmp_df["preempt_times"].mean(), 4)
                self.logger.info(f"VC: {vc_name} | {tmp_df.shape[0]} {type} jobs | Average JCT: {avg_jct} "
                                 f"| Average Queue: {avg_que} | Average Preemption_rate: {avg_preemption_times}")

            tmp_df = df[(df["vc_name"] == vc_name)]
            vc_seq = vc_seq_record[vc_seq_record["vc_name"] == vc_name]
            avg_jct = round(tmp_df["jct"].mean(), 2)
            avg_que = round(tmp_df["queue"].mean(), 2)
            avg_preemption_times = round(tmp_df["preempt_times"].mean(), 4)
            avg_alloc = round(vc_seq["gpu_utilization"].mean(), 4)
            avg_useful = ((self.policy.guar_gpu_time[vc_name] + self.policy.spot_gpu_time[vc_name]) /
                          (self.policy.timespan * vc_seq["total_gpu_num"].max()))

            self.logger.info(f"VC: {vc_name} | Average JCT: {avg_jct} | Average Queue: {avg_que} | Average "
                             f"Preemption_rate: {avg_preemption_times} | Average GPU Allocation: {avg_alloc} "
                             f"| Average Useful GPU-Time Rate: {avg_useful}")

    def pend_job_num_small(self, vc):
        job_num = 0
        for job in self.policy.guar_que_list[vc]:
            if job.__getitem__("gpu_request") < 8:
                job_num += 1
        return job_num
