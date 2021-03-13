#!/bin/bash


#SBATCH --time=30-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DATASET=$1
STATS=$2
SAVE=$3

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 calculate_excess_entropy.py --dataset $DATASET --stats $STATS --save $SAVE --num-proc 4
