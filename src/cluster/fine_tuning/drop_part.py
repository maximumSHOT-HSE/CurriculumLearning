import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import pandas as pd
import random
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-file', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--ratio', type=float, required=True, help='value in [0, 1]')
    parser.add_argument('--type', type=str, required=True, choices=['random', 'begin', 'end'])
    return parser.parse_args()


def extend_dataset(dataset: list, size: int):
    extended = []
    while len(extended) < size:
        extended += dataset
    random.shuffle(extended)
    return extended[:size]


if __name__ == '__main__':
    args = parse_args()

    print(args.dump_file)

    with open(args.dump_file, 'r') as fin:
        dataset = []
        for line in fin:
            dataset.append(line.strip('\x00').strip().split(' '))

    # i, tse, len

    total_size = len(dataset)
    part_size = int(total_size * args.ratio)

    if args.type == 'random':
        random.shuffle(dataset)
        part = dataset[:part_size]
    elif args.type == 'begin':
        part = dataset[:part_size]
    elif args.type == 'end':
        part = dataset[-part_size:]
    else:
        raise NotImplementedError()

    print(f'expected part size = {part_size}, found part_size = {len(part)}')

    extended_part = extend_dataset(part, total_size)

    print(f'expected final size = {total_size}, found final size = {len(extended_part)}')

    with open(args.save, 'w') as fout:
        for entry in extended_part:
            fout.write(f'{int(entry[0])}\n')

