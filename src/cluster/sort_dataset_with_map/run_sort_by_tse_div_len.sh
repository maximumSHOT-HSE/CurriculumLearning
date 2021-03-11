#!/bin/bash


#SBATCH --time=9-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

INPUT=$1
OUTPUT=$2

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 sort_by_tse_div_len.py --input $INPUT --output $OUTPUT --num-proc 4

