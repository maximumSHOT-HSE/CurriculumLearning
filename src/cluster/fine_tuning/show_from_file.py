import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--part', type=str, default='train')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)
    dataset = dataset[args.part]
    # id, metrics, len
    with open(args.file, 'r') as f:
        for line in f:
            x = line.strip('\x00').strip().split(' ')
            id = int(x[0])
            print(f'id = {id}, entry = {dataset[id]}')

