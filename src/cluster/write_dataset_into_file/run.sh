#!/bin/bash


#SBATCH --time=2-10:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 write.py --dataset /home/aomelchenko/datasets/wiki40b_en_tokenized --save wiki40b_en_tokenized_train.txt --part train

