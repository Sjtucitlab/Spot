import time
import os
import argparse
import torch

import pandas as pd
import utils
import cluster


def main(args):
    code_start = time.perf_counter()

    """Logger Setting"""
    log_dir = f"{args.log_dir}/{args.experiment_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir + "/logfile")
    logger = utils.logger_init(file=f"{log_dir}/logfile/main")

    """Infrastructure & Trace Initialization"""
    cluster_trace = args.trace_dir + "/node_info_df.csv"

    CLUSTER = cluster.Cluster(cluster_trace)

    trace_range = ("2024-03-01 00:00:00", "2024-08-31 23:59:59")
    log_range = ("2024-08-01 00:00:00", "2024-08-31 23:59:59")

    log_range = ((pd.Timestamp(log_range[0]) - pd.Timestamp(trace_range[0])).total_seconds(),
                 (pd.Timestamp(log_range[1]) - pd.Timestamp(trace_range[0])).total_seconds())

    trace_df = utils.trace_process(args.trace_dir, trace_range, log_range)

    logger.info(f"Total Job Number in Cluster Training: {len(trace_df)}")

    trace = utils.trace_parser(trace_df)
    args.trace_range = trace_range
    args.log_range = log_range

    """
    Sweep ON: Run All Scheduler Policies in One Experiment
    Sweep OFF: Run Dedicated Scheduler Policy (Default)
    """
    if args.sweep:
        for policy in utils.get_sweep_schedulers():
            utils.simulate_vc(trace, CLUSTER, args, log_dir, policy)
    else:
        utils.simulate_vc(trace, CLUSTER, args, log_dir, args.scheduler)

    logger.info(f"Execution Time: {round(time.perf_counter() - code_start, 2)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulator")
    parser.add_argument("-e", "--experiment-name", default="experiment", type=str, help="Experiment Name")
    parser.add_argument("-t", "--trace-dir", default="./data/experiment", type=str, help="Trace File Directory")
    parser.add_argument("-l", "--log-dir", default="./log", type=str, help="Log Directory")

    parser.add_argument(
        "-s", "--scheduler", default="spot_scheduler", choices=utils.get_available_schedulers(), type=str, help="Scheduler Algorithm"
    )

    parser.add_argument("--colocate", default=1, type=int, help="Whether to enable GPU sharing")
    parser.add_argument("--sweep", action="store_true", default=False, help="Run All Scheduler Policies in One Time")

    # spot guarantee request
    parser.add_argument('--guarantee_hour', type=int, default=[1], help='list of spot guarantee hours')
    parser.add_argument('--guarantee_rate', type=float, default=0.9, help='spot guarantee rate target')
    parser.add_argument('--ckpt_interval', type=int, default=3600, help='spot checkpoint interval(sec)')

    # data loader
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling data")

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--num_covariates', type=int, default=3, help='num of time covariates')
    parser.add_argument('--num_class', type=int, default=3, help='num of input organization classes')
    parser.add_argument('--class_input_dim', type=int, default=[6, 1, 84], help='list of input dims')
    parser.add_argument('--class_output_dim', type=int, default=8, help='output dims of each class')
    parser.add_argument('--attn_hidden_dim', type=int, default=16, help='hidden dim in attention')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00004, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    main(args)
