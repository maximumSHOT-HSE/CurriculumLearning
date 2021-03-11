#!/bin/bash

DATASET=$1
SAVE_DIR=$2
PART=$3

if [[ -d $SAVE_DIR ]]
then
    echo "Save directory exists"
    exit 1
fi

mkdir $SAVE_DIR

#SBATCH --time=10-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 collect_stat_for_tf_idf.py --dataset $DATASET --save $SAVE_DIR --part $PART

