#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 prepare_dataset.py --dataset hyperpartisan_news_detection_50_200_words_tokenized --save hyperpartisan_news_detection_50_200_words_tokenized_with_labels --label-column hyperpartisan

