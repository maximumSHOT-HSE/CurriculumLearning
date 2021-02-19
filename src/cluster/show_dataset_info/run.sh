#!/bin/bash


#SBATCH --time=2-00:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4

module add singularity hpcx/hpcx-ompi
singularity exec --nv ~/containers/container.sif python3 show.py --dataset ../sort_dataset_with_map/wiki40b_en_3M_tokenized_sorted_by_len/ --stats ../collect_stat_from_encoded_dataset/save_stat_3M/save_all_all/ --tokenizer ~/tokenizers/BertTokenizerBase/

