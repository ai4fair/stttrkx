#!/bin/sh

CENV=exatrkx-cpu
CONT=exatrkx-cpu.sif
export EXATRKXDATA=$PWD

# singularity run --nv $CONT -c "conda activate $CENV && python main.py" && exit
singularity run $CONT -c "conda activate $CENV && traintrack configs/pipeline_quickstart.yaml" && exit
