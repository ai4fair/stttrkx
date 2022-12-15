#!/bin/bash

# 1 Node, 1 Task, 1 GPU

#SBATCH -A m3443
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
srun traintrack configs/pipeline_fulltrain.yaml
