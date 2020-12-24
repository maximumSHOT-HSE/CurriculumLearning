#!/bin/bash


#SBATCH --time=10:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 merge_stats.py --prefix save_stat/save_stat_ --save save_stat/save_all_all --check-same-part 0
