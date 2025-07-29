# Source Code for GFS: A Preemption-aware Scheduling Framework for GPU Clusters with Predictive Spot Instance Management

## Overview

This system simulates GPU cluster scheduling with support for spot instances and job preemption. It includes components for cluster management, job scheduling, and performance analysis.

## Project Structure

```text

├── cluster.py          # Cluster and node management implementation
├── job.py              # Job class and trace handling
├── simulator.py        # Main simulation driver
├── utils.py            # Utility functions and trace processing
├── policy/             # Scheduling policy implementations
├── estimator/          # Performance estimators
├── data/               # Input trace data (should contain node_info_df.csv and job_info_df.csv)
├── log/                # Output logs and results
├── requirements.txt    # Python dependencies
└── run.sh              # Execution script
```


## Installation

1. Install dependencies
```python
pip install -r requirements.txt
```

2. Running the Simulation

```python
python simulator.py
```
With custom options:
```python
python simulator.py --experiment-name my_exp --trace-dir ./data/my_trace --scheduler spot_scheduler
```
### Key Arguments
```text
--experiment-name: Name for the experiment (default: "experiment")
--trace-dir: Directory containing trace files (default: "./data/experiment")
--scheduler: Scheduling policy to use (options: "fifo_spot", "spot_scheduler")
```

### Training the Estimator
```python
from estimator.gpu_request_estimator import GPURequestEstimator

estimator = GPURequestEstimator(args)
estimator.train(data, timestamp)
predictions = estimator.test(data, timestamp)
```

## Input Data Requirements
The system requires two CSV files in the trace directory:
1. node_info_df.csv - Cluster node information
2. job_info_df.csv - Job submission trace data
Both datasets are publicly available at https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-spot-gpu

