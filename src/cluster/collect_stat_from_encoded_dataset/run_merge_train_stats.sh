#!/bin/bash


#SBATCH --time=9-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 merge_stats.py --prefix save_stat_3M/save_stat_train --save save_stat_3M/save_all_train 
