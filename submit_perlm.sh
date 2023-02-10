#!/bin/bash

# 1 Node, 1 Task, 1 GPU

#SBATCH -A m3443
#SBATCH -J ctd
#SBATCH -C gpu
#SBATCH -q regular                 # special, early_science
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --signal=SIGUSR1@90        # wall signal

# *** I/O ***
#SBATCH -D .
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.akram@gsi.de

export SLURM_CPU_BIND="cores"
srun traintrack configs/pipeline_fulltrain.yaml

