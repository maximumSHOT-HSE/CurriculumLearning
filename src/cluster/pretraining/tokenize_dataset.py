import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    dataset = datasets \
        .load_from_disk(args.dataset) \
        .map(lambda x: tokenizer(x['text'], truncation=True))

    for i, x in enumerate(dataset['train']):
        print(x)
        break
    print('=============')

    print(dataset)

    dataset.save_to_disk(args.save)

