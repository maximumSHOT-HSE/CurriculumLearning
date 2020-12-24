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
    parser.add_argument('--prefix', type=str, help='Prefix of directories names from which partial statistics will be retrieved for merging', required=True)
    parser.add_argument('--save', type=str, help='Path to the EMPTY directory, where statistics will be saved', required=True)
    parser.add_argument('--check-same-part', type=int, default=1, help='Whether to check similarity of dataset parts or not')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    roots = list(sorted(Path().rglob(f'{args.prefix}*')))

    len_counter = Counter()
    last_counter =  Counter()
    single_counter = Counter()
    pair_counter = Counter()

    save_root = Path(args.save)

    all_paths = {
        'len_counter.bin': len_counter,
        'last_counter.bin': last_counter,
        'single_counter.bin': single_counter,
        'pair_counter.bin': pair_counter
    }

    config = None

    for root in roots:
        with open(root / 'config.json', 'r') as f:
            c = json.load(f)
            if config is None:
                config = c
            else:
                assert config['hash-base'] == c['hash-base']
                assert config['tokenizer'] == c['tokenizer']
                assert config['dataset'] == c['dataset']
                if args.check_same_part:
                    assert config['part'] == c['part']

        assert config is not None

        tmp_len_counter = Counter()
        tmp_last_counter =  Counter()
        tmp_single_counter = Counter()
        tmp_pair_counter = Counter()

        paths = {
            'len_counter.bin': tmp_len_counter,
            'last_counter.bin': tmp_last_counter,
            'single_counter.bin': tmp_single_counter,
            'pair_counter.bin': tmp_pair_counter
        }

        for path in paths.keys():
            tmp_counter = paths[path]
            all_counter = all_paths[path]
            with open(root / path, 'rb') as f:
                tmp_counter = pickle.load(f)
                all_counter += tmp_counter

    all_config = {
        'hash-base': config['hash-base'],
        'tokenizer': config['tokenizer'],
        'dataset': config['dataset'],
    }

    if args.check_same_part:
        all_config['part'] = config['part']

    print(all_config)

    for path, counter in all_paths.items():
        print(path, len(counter))
        with open(save_root / path, 'wb') as f:
            pickle.dump(counter, f)

    with open(save_root / 'config.json', 'w') as f:
        json.dump(all_config, f)

