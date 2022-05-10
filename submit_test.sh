#!/bin/bash

# Export Variables
# export LUSTRE_HOME=/lustre/$(id -g -n)/$USER
# export EXATRKX_DATA=$LUSTRE_HOME/ctd2022

#SBATCH -A panda -J CTD22 -p main -c32 -n1 -N1 --gres=gpu:tesla:1 --mem=32768 -t 0:30:00
srun singularity run --nv $LUSTRE_HOME/containers/gpu_stttrkx.sif -c "conda activate exatrkx && python test.py"
