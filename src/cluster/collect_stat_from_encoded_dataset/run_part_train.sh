#!/bin/bash

START=$1
END=$2
SAVE_DIR=$3
PART=$4
DATASET=$5

if [[ -d $SAVE_DIR ]]
then
    echo "Save directory exists"
    exit 1
fi

mkdir $SAVE_DIR

#SBATCH --time=3-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 collect_stat.py --dataset $DATASET --config config.json --save $SAVE_DIR --start $START --end $END --tokenizer ~/tokenizer_cased/ --part $PART
