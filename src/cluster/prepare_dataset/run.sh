#!/bin/bash


#SBATCH --time=10:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 collect_stat.py --dataset ~/datasets/wiki40b_en_encoded_cased/ --config config.json --save tmp_save/ --start 0 --end 10000 --tokenizer ~/tokenizer_cased/ --part train 
