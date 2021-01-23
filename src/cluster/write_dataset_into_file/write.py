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
    parser.add_argument('--save', type=str, help='Path to the file to save', required=True)
    parser.add_argument('--part', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset)

    with open(args.save, 'w') as f:
        for x in tqdm(dataset[args.part]):
            f.write(' '.join(map(str, x['input_ids'])) + '\n')
