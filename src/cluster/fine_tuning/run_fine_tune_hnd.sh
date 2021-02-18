#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4

DIR='hyperpartisan_news_detection/hnd_50_200_words_baseline'

mkdir $DIR

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 fine_tune.py --dataset hyperpartisan_news_detection/hyperpartisan_news_detection_50_200_words_tokenized/ --tokenizer ~/tokenizers/BertTokenizerBaseCased/ --model ~/models/pretrained_bert_base_cased/ --output-dir "$DIR/out" --logging-dir "$DIR/logs"

