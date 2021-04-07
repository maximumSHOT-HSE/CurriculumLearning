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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    stat_root = Path(args.stats)
    save_root = Path(args.save)

    paths = {
        'single_counter.bin': Counter(),
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

    F_single = paths['single_counter.bin']

    occur = Counter()
    total = 0

    for key in F_single:
        x_i = key // BASE
        oc = F_single[key]
        occur[x_i] += oc
        total += oc

    for key in occur:
        occur[key] /= total

    with open(save_root / 'word_freq.bin', 'wb') as f:
        pickle.dump(occur, f)

