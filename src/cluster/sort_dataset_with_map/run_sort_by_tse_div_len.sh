#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

INPUT='../sort_dataset_by_column/wiki40b_encoded_cased_sorted_by_tse/'
OUTPUT='wiki40b_encoded_cased_sorted_by_tse_div_len'

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 sort_by_tse_div_len.py --input $INPUT --output $OUTPUT --num-proc 4
