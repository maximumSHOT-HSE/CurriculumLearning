import argparse
import datasets
from transformers import BertTokenizer
from transformers import EvaluationStrategy
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--old-column', type=str, required=True)
    parser.add_argument('--new-column', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)
    dataset.rename_column_(args.old_column, args.new_column)
    dataset.save_to_disk(args.save)

