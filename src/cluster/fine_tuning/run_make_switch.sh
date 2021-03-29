#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

INPUT=$1
BATCH_SIZE=$2
STEPS=$3
OUTPUT=$4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 make_switch.py --input $INPUT --batch-size $BATCH_SIZE --steps $STEPS --output $OUTPUT

