import sys
import os
import logging
import pandas as pd
from job import Job, Trace
from policy import (
    FifoWithSpot,
    SpotScheduler,
    Chronus,
    Lyra,
    FGD
)

sys.path.append("..")


def simulate_vc(trace, cluster, args, log_dir, policy):
    if not os.path.exists(f"{log_dir}/{cluster.cluster_name}"):
        os.makedirs(f"{log_dir}/{cluster.cluster_name}")
    logger = logger_init(file=f"{log_dir}/{cluster.cluster_name}/{policy}_out")
    if policy == "fifo_spot":
        scheduler = FifoWithSpot(trace, cluster, args, log_dir, logger)
    elif policy == "spot_scheduler":
        scheduler = SpotScheduler(trace, cluster, args, log_dir, logger)
    elif policy == "chronus":
        scheduler = Chronus(trace, cluster, args, log_dir, logger)
    elif policy == "lyra":
        scheduler = Lyra(trace, cluster, args, log_dir, logger)
    elif policy == "FGD":
        scheduler = FGD(trace, cluster, args, log_dir, logger)
    scheduler.simulate()
    logger.info(f"Finish {cluster.cluster_name} by {policy} Policy")
    return True


def get_available_schedulers():
    return ["fifo_spot", "spot_scheduler"]


def get_sweep_schedulers():
    return ["fifo_spot", "spot_scheduler"]


# 任务trace数据
def trace_process(dir, date_range, log_range):
    df = pd.read_csv(
        dir + "/job_info_df.csv",
        parse_dates=["submit_time"],
        usecols=["job_name", "organization", "gpu_model", "worker_num", "gpu_request",
                 "cpu_request", "submit_time", "duration", "job_type"],
    )
    df["submit_time"] = df["submit_time"].astype(float)
    df = df[(df["job_type"] == "HP") | ((df["submit_time"] >= log_range[0]) & (df["submit_time"] <= log_range[1]))]

    # df = df[df["gpu_model"] == "A100-SXM4-80GB"]

    df["job_index"] = df.index
    df["vc_name"] = df["gpu_model"]
    df.rename(columns={"job_type": "type"}, inplace=True)

    # Consider gpu jobs only
    df = df[df["gpu_request"] > 0]

    df["org_key"] = df["gpu_model"] + '&&cluster&&' + df["organization"].astype(str)
    df["remain"] = df["duration"]
    df[["start_time", "end_time"]] = sys.maxsize
    df[["ckpt_times", "preempt_times", "queue", "jct"]] = 0
    df["guarantee_request"] = 1
    df["status"] = None

    # Slicing simulation part
    end = (pd.Timestamp(date_range[1]) - pd.Timestamp(date_range[0])).total_seconds()
    df = df[(df["submit_time"] >= 0) & (df["submit_time"] <= end)]

    df.sort_values(by="submit_time", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def trace_parser(df):
    trace = Trace()

    for _, series in df.iterrows():
        trace.append_job(Job(series))
    trace.sort_jobs("submit_time")
    return trace


def logger_init(file):
    logger = logging.getLogger()
    handler_file = logging.FileHandler(f"{file}.log", "w")
    handler_stream = logging.StreamHandler()  # sys.stdout

    logger.setLevel(logging.INFO)
    handler_file.setLevel(logging.INFO)
    handler_stream.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(processName)s | %(message)s", datefmt="%Y %b %d %H:%M:%S")
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)

    return logger


def cluster_concatenate(policy, placer, log_dir, cluster):
    prefix = f"{policy}_{placer}"
    if not os.path.exists(log_dir + "/all"):
        os.mkdir(log_dir + "/all")

    """Log"""
    cluster_log = pd.read_csv(f"{log_dir}/{cluster.cluster_name}/{prefix}_{cluster.cluster_name}_log.csv")
    cluster_log.sort_values(by="submit_time", inplace=True)
    cluster_log.to_csv(f"{log_dir}/all/{prefix}_all_log.csv", index=False)

    """Seq"""
    cluster_seq = pd.read_csv(f"{log_dir}/{cluster.cluster_name}/{prefix}_{cluster.cluster_name}_seq.csv")

    cluster_seq.dropna(inplace=True)
    cluster_seq = cluster_seq.astype(int)
    cluster_seq["gpu_utilization"] = (
            (cluster_seq["total_gpu_num"] - cluster_seq["idle_gpu_num"]) / cluster_seq["total_gpu_num"]
    ).round(3)
    cluster_seq.to_csv(f"{log_dir}/all/{prefix}_all_seq.csv", index=False)


def cluster_analysis(placer, log_dir, dir):
    """Generate Algorithm Comparsion CSV"""
    # ignore_warm_up = start_ts + 7*24*3600

    vc_df = pd.read_csv(dir + "/raw_node_info_df.csv")
    vcs = list(vc_df.vc_name.unique())
    vcs.append("all")

    files = os.listdir(f"{log_dir}/all")
    prefix = set()
    for file in files:
        policy = file.split("_")[0]
        placer = file.split("_")[1]
        prefix.add(f"{policy}_{placer}")
    prefix_list = sorted(list(prefix))

    jct_avg = pd.DataFrame()
    que_avg = pd.DataFrame()
    for prefix in prefix_list:
        for vc in vcs:
            vc_log = pd.read_csv(f"{log_dir}/{vc}/{prefix}_{vc}_log.csv")
            # vc_log = vc_log[vc_log['submit_time'] > ignore_warm_up]
            jct_avg.at[vc, prefix] = vc_log["jct"].mean()
            que_avg.at[vc, prefix] = vc_log["queue"].mean()

    jct_avg = jct_avg.astype(int)
    que_avg = que_avg.astype(int)
    jct_avg.to_csv(f"{log_dir}/jct_avg_{placer}.csv")
    que_avg.to_csv(f"{log_dir}/que_avg_{placer}.csv")
