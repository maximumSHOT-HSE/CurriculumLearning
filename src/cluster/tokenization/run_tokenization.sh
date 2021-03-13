#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python tokenization_experiments.py --vocab_size $1

