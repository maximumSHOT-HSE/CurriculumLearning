#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

DUMP_FILE=$1
SAVE=$2
RATIO=$3
TYPE=$4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 drop_part.py --dump-file $DUMP_FILE --save $SAVE --ratio $RATIO --type $TYPE

