#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DATASET=$1
SEED=$2
TOKENIZER=$3
MODEL=$4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 evaluate.py --dataset $DATASET --tokenizer $TOKENIZER --model $MODEL --seed $SEED

