#!/bin/bash


#SBATCH --time=9-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

INPUT='/home/aomelchenko/datasets/wiki40b_en_3M_tokenized'
OUTPUT='wiki40b_en_3M_tokenized_sorted_by_len'

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 sort_by_len.py --input $INPUT --output $OUTPUT --num-proc 4
