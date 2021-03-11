#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DATASET=$1
SAVE=$2

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 make_labels.py --dataset $DATASET --save $SAVE 

