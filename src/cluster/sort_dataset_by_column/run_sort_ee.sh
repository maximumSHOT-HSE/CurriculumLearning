#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

INPUT='../calculate_excess_entropy/wiki40b_en_3M_tokenized_with_ee/'
COLUMN='excess_entropy'
OUTPUT='wiki40b_en_3M_sorted_by_ee'

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 sort.py --input $INPUT --column $COLUMN --output $OUTPUT 
