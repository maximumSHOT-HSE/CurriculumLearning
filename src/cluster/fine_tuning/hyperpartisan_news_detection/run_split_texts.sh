#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DATASET=$1
SAVE=$2
MIN_LENGTH=$3
MAX_LENGTH=$4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 split_texts.py --dataset $DATASET --save $SAVE --max-length $MAX_LENGTH --min-length $MIN_LENGTH 

