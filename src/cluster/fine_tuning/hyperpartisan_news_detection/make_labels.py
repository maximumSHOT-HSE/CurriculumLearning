import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets \
        .load_from_disk(args.dataset) \
        .map(lambda x: {'labels': int(x['hyperpartisan'])})

    dataset.remove_columns_('hyperpartisan')

    print(dataset)

    for i, x in enumerate(dataset['train']):
        print(x)
        if i >= 5:
            break

    dataset.save_to_disk(args.save)

