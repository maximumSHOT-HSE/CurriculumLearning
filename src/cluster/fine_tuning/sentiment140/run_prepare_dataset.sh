#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

SAVE=$1

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 prepare_dataset.py --csv-file data.csv --save $SAVE --tokenizer ~/tokenizers/BertTokenizerBaseCased/

