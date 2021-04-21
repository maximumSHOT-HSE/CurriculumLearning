#!/bin/bash


#SBATCH --time=7-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

CONFIG=$1
SAVE=$2

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 merge.py --config $CONFIG --save $SAVE
