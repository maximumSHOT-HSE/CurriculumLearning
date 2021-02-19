#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8


module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 eval.py --dataset sentiment140/sentiment140_prepaired_tokenized_LABELS --tokenizer ~/tokenizers/BertTokenizerBaseCased/ --model   sentiment140/baseline/out/checkpoint-25000/

