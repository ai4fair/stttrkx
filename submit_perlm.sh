#!/bin/bash

# 1 Node, 1 Task, 1 GPU

#SBATCH -A m3443_g
#SBATCH -J ctd
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --requeue                  # automatically requeue slurm jobs
#SBATCH --signal=SIGUSR1@90        # signal to pytorch-lighting about job termination

# *** I/O ***
#SBATCH -D .
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.akram@gsi.de

export SLURM_CPU_BIND="cores"

mkdir -p logs
eval "$(conda shell.bash hook)"
conda activate exatrkx-cori
export EXATRKX_DATA=$PSCRATCH

srun traintrack $HOME/ctd2022/configs/pipeline_fulltrain.yaml

