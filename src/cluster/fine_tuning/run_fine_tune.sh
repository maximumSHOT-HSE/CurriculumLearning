#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATASET=$1
TRAINER=$2
DIR=$3

mkdir $DIR

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 fine_tune.py --dataset $DATASET --tokenizer ~/tokenizers/BertTokenizerBaseCased/ --model ~/models/pretrained_bert_base_cased/ --output-dir "$DIR/out" --logging-dir "$DIR/logs" --trainer $TRAINER

