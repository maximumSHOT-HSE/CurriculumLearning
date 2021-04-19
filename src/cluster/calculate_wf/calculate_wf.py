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
from math import log2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with encoded dataset', required=True)
    parser.add_argument('--stats', type=str, help='Path to the directory with statistics', required=True)
    parser.add_argument('--save', type=str, help='Path to the where dataset with calculated tse will be saved', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset)

    print(dataset)

    stat_root = Path(args.stats)

    freq = Counter()

    with open(stat_root / 'word_freq.bin', 'rb') as f:
        freq = pickle.load(f)

    print(freq)
    print(f'len = {len(freq)}')

    tokens = list(freq)
    tokens.sort(key=lambda x: -freq[x])

    rank = Counter()
    for i, x in enumerate(tokens):
        rank[x] = i

    print(rank)

    def calculate_max_wf_rank(xs):
        return max(rank[x] for x in xs)

    def calculate_avg_wf_rank(xs):
        return sum(rank[x] for x in xs) / len(xs)

    def calculate_minus_log_likelihood(xs):
        return -sum(log2(freq[x] + 1e-18) for x in xs)

    dataset.map(lambda x: {
        'max_wf_rank': calculate_max_wf_rank(x['input_ids']),
        'avg_wf_rank': calculate_avg_wf_rank(x['input_ids']),
        'minus_log_likelihood': calculate_minus_log_likelihood(x['input_ids'])
    }).save_to_disk(args.save)

