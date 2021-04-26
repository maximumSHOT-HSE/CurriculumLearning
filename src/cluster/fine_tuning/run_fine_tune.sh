#!/bin/bash


#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4

DATASET=$1
TRAINER=$2
DIR=$3
SEED=$4
FROM_FILE=$5
MODEL=$6
WARMUP_STEPS=$7
MAX_STEPS=$8

mkdir $DIR

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 fine_tune.py --dataset $DATASET --tokenizer ~/tokenizers/BertTokenizerBaseCased/ --model $MODEL --output-dir "$DIR/out" --logging-dir "$DIR/logs" --trainer $TRAINER --seed $SEED --from-file $FROM_FILE --warmup-steps $WARMUP_STEPS --max-steps $MAX_STEPS

