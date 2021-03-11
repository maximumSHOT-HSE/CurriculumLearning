#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DUMP_FILE=$1
BATCH_SIZE=$2
SAVE=$3

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 sort_by_len_merge.py --dump-file $DUMP_FILE --batch-size $BATCH_SIZE --save $SAVE

