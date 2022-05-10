#!/bin/bash

# Export Variables
export LUSTRE_HOME=/lustre/$(id -g -n)/$USER
export SLURM_SUBMIT_DIR=$LUSTRE_HOME"/ctd2022"
export EXATRKX_DATA=$SLURM_SUBMIT_DIR

export SLURM_WORKING_DIR=$LUSTRE_HOME"/ctd2022/logs"
mkdir -p $SLURM_WORKING_DIR;

CENV=exatrkx
CONT=gpu_stttrkx.sif

# One-liner Meta Commands
#SBATCH -A panda -J Test -t 00:30 -p main -N1 -c16 --gres=gpu:tesla:1
srun -D $SLURM_WORKING_DIR -o '%x-%j.out' -e '%x-%j.err' -- singularity run --nv $LUSTRE_HOME/containers/$CONT -c "conda activate $CENV && python test.py"
