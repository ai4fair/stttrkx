#!/bin/bash

# Export Variables
export LUSTRE_HOME=/lustre/$(id -g -n)/$USER
export SLURM_SUBMIT_DIR=$LUSTRE_HOME"/ctd2022"
export EXATRKX_DATA=$SLURM_SUBMIT_DIR
export SLURM_WORKING_DIR=$LUSTRE_HOME"/ctd2022/logs"
mkdir -p $SLURM_WORKING_DIR;

CENV= exatrkx
CONT=gpu_stttrkx.sif

srun -A panda -J Test -t 00:30 -o '%x-%j.out' -e '%x-%j.err' -D $SLURM_WORKING_DIR -- singularity run --nv $LUSTRE_HOME/containers/$CONT -c "conda activate $CENV && python test.py"
