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
from metrics import excess_entropy_fast, calculate_entropy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with encoded dataset', required=True)
    parser.add_argument('--stats', type=str, help='Path to the directory with statistics', required=True)
    parser.add_argument('--save', type=str, help='Path to the where dataset with calculated excess entropy will be saved', required=True)
    parser.add_argument('--num-proc', type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset)

    print(dataset)

    stat_root = Path(args.stats)

    paths = {
        'len_counter.bin': Counter(),
        'last_counter.bin': Counter(),
        'single_counter.bin': Counter(),
        'pair_counter.bin': Counter()
    }

    for path in paths.keys():
        with open(stat_root / path, 'rb') as f:
            paths[path] = pickle.load(f)

    with open(stat_root / 'config.json', 'r') as f:
        config = json.load(f)

    print(config)

    BASE = int(config['hash-base'])
    BASE_SQR = BASE ** 2
    print('base', BASE)

    max_len = max(paths['len_counter.bin'].keys())
    print('max_len', max_len)

    F_pos = np.zeros(max_len + 1, dtype=int)

    for i, x in paths['len_counter.bin'].items():
        if i > 0:
            F_pos[i - 1] += x

    for n in range(max_len, 0, -1):
        F_pos[n - 1] += F_pos[n]

    F_last = paths['last_counter.bin']
    F_single = paths['single_counter.bin']
    F_pair = paths['pair_counter.bin']

    def H_single(i, x_i):
        p = F_single[i + x_i * BASE] / F_pos[i]
        return calculate_entropy([p, 1 - p])

    def H_pair(i, x_prv, x_cur):
        T = F_pos[i]
        c11 = F_pair[(i - 1) + x_prv * BASE + x_cur * BASE_SQR]
        c01 = F_single[i + x_cur * BASE] - c11
        c10 = F_single[(i - 1) + x_prv * BASE] - c11 - F_last[(i - 1) + x_prv * BASE]
        c00 = T - c11 - c01 - c10
        assert c00 >= 0, f'c00 = {c00}, c11 = {c11}, c01 = {c01}, c10 = {c10}, T = {T}, i = {i}, x_prv = {x_prv}, x_cur = {x_cur}'
        p = (c10 + c11) / T
        return calculate_entropy(np.array([c00, c01, c10, c11], dtype=float) / T) - calculate_entropy(np.array([p, 1 - p], dtype=float))

    dataset.map(lambda x: {'excess_entropy': excess_entropy_fast(x['input_ids'], H_single, H_pair)}, num_proc=args.num_proc).save_to_disk(args.save)
