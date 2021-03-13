import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--test-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    parts = []
    for part in dataset:
        parts.append(dataset[part])
    dataset = datasets.concatenate_datasets(parts)

    dataset = dataset.shuffle(seed=args.seed).train_test_split(test_size=args.test_ratio)

    print('AFTER RESPLIT')
    print(dataset)

    dataset.save_to_disk(args.save)

