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
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with encoded dataset', required=True)
    parser.add_argument('--save', type=str, help='Path to the where dataset with calculated tse will be saved', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset)

    print(dataset)

    def calculate_TPW(sentence, tokens):
        n_tokens = sum(1 for x in tokens if int(x) > 0)
        n_words = len(list(filter(lambda x: len(x) > 0, re.split('[ ,.?!:;]+', sentence))))
        return float(n_tokens) / float(n_words + 1)

    dataset.map(lambda x: {
        'TPW': calculate_TPW(x['text'], x['input_ids'])
    }).save_to_disk(args.save)

