import argparse
import datasets
from transformers import BertTokenizer
from transformers import EvaluationStrategy
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import hashlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    print(dataset)
    print()

    for part in dataset:
        texts = dataset[part]['text']
        lengths = list(map(len, texts))
        hashes = list(map(lambda t: hashlib.md5(t.encode('utf-8')).hexdigest(), texts))
        metric_values = dataset[part][args.metric]
        df = pd.DataFrame.from_dict({'hash': hashes, 'length': lengths, args.metric: metric_values})
        df.to_csv(f'{args.save}_{part}.csv', index=False)
