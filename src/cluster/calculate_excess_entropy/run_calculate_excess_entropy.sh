#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 calculate_excess_entropy.py --dataset ~/datasets/wiki40b_en_encoded_cased/ --stats ../collect_stat_from_encoded_dataset/save_stat/save_all_all/ --save wiki40b_en_encoded_cased_with_ee
