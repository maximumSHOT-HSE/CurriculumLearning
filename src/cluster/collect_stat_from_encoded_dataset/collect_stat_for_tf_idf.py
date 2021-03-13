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
    parser.add_argument('--dataset', type=str, help='Path to the directory with dataset', required=True)
    parser.add_argument('--save', type=str, help='Path to the directory, where statistics will be saved', required=True)
    parser.add_argument('--part', type=str, help='Part of the dataset (train/validation/test)', required=True, choices=['train', 'validation', 'test'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset)

    config = {}

    config['dataset'] = args.dataset
    config['dataset'] = args.dataset
    config['part'] = args.part

    docs_counter = Counter()

    print(dataset[args.part])
    print(config)

    for it, item in tqdm(enumerate(dataset[args.part])):
        x = np.array(item['input_ids'])
        unique_x = np.unique(x)
        docs_counter += Counter(unique_x)

    root = Path(args.save)

    with open(root / 'docs_counter.bin', 'wb') as f:
        pickle.dump(docs_counter, f)

    with open(root / 'config.json', 'w') as f:
        json.dump(config, f)

