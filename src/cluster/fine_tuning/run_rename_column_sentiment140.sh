#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 rename_column.py --dataset sentiment140/sentiment140_prepaired_tokenized --old-column label --new-column labels --save sentiment140/sentiment140_prepaired_tokenized_LABELS

