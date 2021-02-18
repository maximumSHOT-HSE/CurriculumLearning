#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

INPUT='wiki40b_encoded_cased_sorted_by_tse'

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 test.py --input $INPUT
