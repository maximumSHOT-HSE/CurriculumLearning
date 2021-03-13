import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)
    print(dataset)

    for i, x in enumerate(dataset):
        print(i, x)
        if i >= 5:
            break

