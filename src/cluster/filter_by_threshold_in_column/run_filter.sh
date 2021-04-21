#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

INPUT=$1
COLUMN=$2
OUTPUT=$3
THRESHOLD=$4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 filter.py --input $INPUT --column $COLUMN --output $OUTPUT --threshold $THRESHOLD

