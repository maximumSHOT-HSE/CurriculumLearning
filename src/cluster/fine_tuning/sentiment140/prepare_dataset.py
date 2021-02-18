import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--test-ratio', type=float, default=0.005)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = pd.read_csv(args.csv_file, encoding='ISO-8859-1', header=None)

    print(dataset.describe())
    print(dataset.columns)
    print(len(dataset))

    labels = dataset[0]
    texts = dataset[5]

    print(set(labels))
    labels = [x // 4 for x in labels]
    print(set(labels))

    print(len(labels), len(texts))

    for i, (l, t) in enumerate(zip(labels, texts)):
        print(type(l), type(t), l, t)
        print('=============')
        if i >= 5:
            break

    dataset = datasets.Dataset.from_dict({'text': texts, 'label': labels}).train_test_split(test_size=args.test_ratio)

    print(dataset)

    for part in dataset:
        print('part', part)
        for x in dataset[part]:
            print(x)
            break

    dataset.save_to_disk(args.save)

