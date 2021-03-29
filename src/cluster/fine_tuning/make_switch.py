import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import pandas as pd
import random
import math
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(args.input)

    with open(args.input, 'r') as fin:
        dataset = []
        for line in fin:
            dataset.append(line.strip('\x00').strip())

    # i, tse, len
    prefix_size = args.batch_size * args.steps
    prefix = deepcopy(dataset[:prefix_size])
    print(f'prefix size = {len(prefix)} = {prefix_size}')
    print(f'before prefix = {dataset[:prefix_size]}')
    random.shuffle(dataset)
    print(f'after shuffle prefix = {dataset[:prefix_size]}')
    dataset[:prefix_size] = prefix
    print(f'final prefix = {dataset[:prefix_size]}')

    with open(args.output, 'w') as fout:
        for x in dataset:
            fout.write(f'{x}\n')

