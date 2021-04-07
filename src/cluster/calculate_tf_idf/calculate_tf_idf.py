from transformers import BertForSequenceClassification
from tokenizers import normalizers, pre_tokenizers, Tokenizer, models, trainers
import datasets
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
import pickle
import argparse
import json
from pathlib import Path
from metrics import TSE_fast, calculate_entropy, TF_IDF


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with encoded dataset', required=True)
    parser.add_argument('--stats', type=str, help='Path to the directory with statistics', required=True)
    parser.add_argument('--save', type=str, help='Path to the where dataset with calculated tse will be saved', required=True)
    parser.add_argument('--num-proc', type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset)

    print(dataset)

    stat_root = Path(args.stats)

    print(f'stat_root = {stat_root}')

    with open(stat_root / 'docs_counter.bin', 'rb') as f:
        docs_counter = pickle.load(f)

    with open(stat_root / 'config.json', 'r') as f:
        config = json.load(f)

    print(config)

    def calc_idf(x):
        return np.log()

    size = len(dataset['train'])
    print(f'size = {size}')
    def calc_idf(x):
        return np.log(size / (1 + docs_counter[x]))
    dataset.map(lambda x: {'tf-idf': TF_IDF(x['input_ids'], calc_idf)}, num_proc=args.num_proc).save_to_disk(args.save)
