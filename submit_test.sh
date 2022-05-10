#!/bin/sh
#SBATCH -A panda -J Test -t 12:30 -p main -N1 -c16 --gres=gpu:tesla:1 --mail-type=END --mail-user=a.akram@gsi.de

export LUSTRE_HOME=/lustre/$(id -g -n)/$USER
export EXATRKX_DATA=$LUSTRE_HOME/ctd2022

CENV=exatrkx
CONT=$LUSTRE_HOME/containers/gpu_stttrkx.sif

#singularity run --nv $CONT -c "conda activate $CENV && python test.py"
singularity run --nv $CONT -c "conda activate $CENV && traintrack configs/pipeline_quickstart.yaml"
