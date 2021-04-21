#!/bin/bash


#SBATCH --time=7-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DATASET=$1
MODEL=$2
SAVE=$3
METRIC=$4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 predict.py --dataset $DATASET --tokenizer ~/tokenizers/BertTokenizerBaseCased/ --model $MODEL --save $SAVE --metric $METRIC
