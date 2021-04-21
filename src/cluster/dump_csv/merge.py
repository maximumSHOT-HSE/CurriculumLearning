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
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    merged_df = None

    with open(args.config, 'r') as f:
        for line in f:
            path = line.strip()
            df = pd.read_csv(path)
            if merged_df is None:
                mergd_df = df
                continue
            print(df)
            print(merged_df)
            merged_df = merged_df.merge(df, how='inner', on=['hash', 'length'])
            print(merged_df)
            exit(0)
