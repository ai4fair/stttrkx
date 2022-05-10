#!/bin/sh
#SBATCH -A panda -J Test -t 00:30 -p main -N1 -c16 --gres=gpu:tesla:1 -D . -o '%x-%j.out' -e '%x-%j.err'

export LUSTRE_HOME=/lustre/$(id -g -n)/$USER
export EXATRKX_DATA=$LUSTRE_HOME/ctd2022

CENV=exatrkx
CONT=gpu_stttrkx.sif
#singularity run --nv $LUSTRE_HOME/containers/$CONT -c "conda activate $CENV && traintrack configs/pipeline_quickstart.yaml"
singularity run --nv /lustre/pbar/aakram/containers/$CONT -c "conda activate $CENV && python test.py"

