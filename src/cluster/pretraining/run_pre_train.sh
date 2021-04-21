#!/bin/bash


#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATASET=$1
TRAINER=$2
DIR=$3
SEED=$4
FROM_FILE=$5
TOKENIZER=$6
MODEL=$7
MLM_PROB=$8
REVERSE=$9

mkdir $DIR

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 pre_train.py --dataset $DATASET --tokenizer $TOKENIZER --model $MODEL --output-dir "$DIR/out" --logging-dir "$DIR/logs" --trainer $TRAINER --seed $SEED --from-file $FROM_FILE --mlm-prob $MLM_PROB --reverse $REVERSE

