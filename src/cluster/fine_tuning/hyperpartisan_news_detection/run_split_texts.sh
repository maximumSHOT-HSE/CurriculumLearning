#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 split_texts.py --dataset hyperpartisan_news_detection_cleaned --save hyperpartisan_news_detection_50_200_words --max-length 200 --min-length 50

