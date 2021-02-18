#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4

DIR='sentiment140/debug'

mkdir $DIR

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 fine_tune.py --dataset sentiment140/sentiment140_prepaired_tokenized --tokenizer ~/tokenizers/BertTokenizerBaseCased/ --model ~/models/pretrained_bert_base_cased/ --output-dir "$DIR/out" --logging-dir "$DIR/logs"

