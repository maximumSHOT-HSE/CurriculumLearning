#!/bin/bash


#SBATCH --time=9-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

PREFIX=$1
SAVE=$2
CHECK_SAME_PART=$3

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 merge_stats.py --prefix $PREFIX --save $SAVE --check-same-part $CHECK_SAME_PART
