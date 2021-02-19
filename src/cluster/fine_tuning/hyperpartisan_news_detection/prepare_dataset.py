import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--label-column', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets \
        .load_from_disk(args.dataset) \
        .map(lambda x: {'label': int(x[args.label_column])})

    dataset.remove_columns_([args.label_column])

    for i, x in enumerate(dataset['train']):
        print(x)
        break
    print('=============')
    for i, x in enumerate(dataset['test']):
        print(x)
        break
    print('=============')

    dataset.save_to_disk(args.save)

