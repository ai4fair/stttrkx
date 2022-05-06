#!/bin/sh

CENV=exatrkx
CONT=gpu_stttrkx.sif
CONT=cpu_stttrkx.sif

export EXATRKXDATA=$PWD

singularity run $CONT -c "conda activate $CENV && traintrack configs/pipeline_quickstart.yaml"
