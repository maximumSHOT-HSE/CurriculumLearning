#!/bin/bash


#SBATCH --time=00:01:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1

DATASET=$1

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 print_dataset_info.py --dataset $DATASET

