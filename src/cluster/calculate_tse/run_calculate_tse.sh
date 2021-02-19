#!/bin/bash


#SBATCH --time=30-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 calculate_tse.py --dataset ~/datasets/wiki40b_en_reduced_tokenized --stats ../collect_stat_from_encoded_dataset/save_stat_reduced/save_all_all/ --save wiki40b_en_reduced_with_tse --num-proc 4
