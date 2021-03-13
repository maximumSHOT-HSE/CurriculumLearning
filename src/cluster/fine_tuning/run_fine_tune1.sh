#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=MF_base
#SBATCH --output=MF_base.log

TOKENIZER="/home/aomelchenko/tokenizers/BertTokenizerBase"
DATASET='/home/aomelchenko/datasets/wiki40b_en_3M_tokenized'
TRAINER='hyperbole'
DIR='Logs/MF_base'
SEED=42
MODEL='/home/aomelchenko/BertBaseConfigReduced'

mkdir $DIR

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 max_bert.py --dataset $DATASET --tokenizer $TOKENIZER --model $MODEL --output-dir "$DIR/out" --logging-dir "$DIR/logs" --trainer $TRAINER --seed $SEED

