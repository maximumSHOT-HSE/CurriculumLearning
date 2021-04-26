#!/bin/bash


#SBATCH --time=7-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DATASET=$1
MAX_NOISE_LEVEL=$2
SAVE=$3

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 make_noisy_input.py --dataset $DATASET --max-noise-level $MAX_NOISE_LEVEL --save $SAVE

