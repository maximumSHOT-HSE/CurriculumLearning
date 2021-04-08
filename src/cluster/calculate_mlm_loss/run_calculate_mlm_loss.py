#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DATASET=$1
TOKENIZER=$2
MODEL=$3
SAVE=$4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 calculate_mlm_loss.py --dataset $DATASET --tokenizer $TOKENIZER --model $MODEL --save $SAVE

