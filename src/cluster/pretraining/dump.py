import argparse
import datasets
from transformers import BertTokenizer
from transformers import EvaluationStrategy
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--column', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    for part in dataset:
        filename = f'{args.save}.{part}'
        with open(filename, 'w') as fout:
            for i, x in enumerate(dataset[part]):
                l = sum(1 for c in x['input_ids'] if int(c) > int(0))
                fout.write(f'{i} {x[args.column]} {l}\n')

