#!/bin/sh
#SBATCH -A panda -J Test -t 00:30 -p main -N1 -c16 --gres=gpu:tesla:1 -D . -o '%x-%j.out' -e '%x-%j.err'

CENV=exatrkx
CONT=gpu_stttrkx.sif
singularity run --nv /lustre/pbar/aakram/containers/$CONT -c "conda activate $CENV && python test.py"
