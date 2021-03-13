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
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


def group_into_batches(xs: list, batch_size: int):
    batches = []
    n = len(xs)
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        # [i, j)
        batches.append(xs[i: j])
    return batches


if __name__ == '__main__':
    args = parse_args()

    print(args.dump_file)

    with open(args.dump_file, 'r') as fin:
        dataset = []
        for line in fin:
            dataset.append(line.strip('\x00').strip().split(' '))

    # i, tse, len

    dataset.sort(key=lambda x: int(x[2]))
    batches = group_into_batches(dataset, math.ceil(len(dataset) / args.batch_size))

    sum_len = 0
    for batch in batches:
        sum_len += len(batch)

    print(f'sum_len = {sum_len}')
    print(f'total len = {len(dataset)}')
    print(f'batches count = {len(batches)}')

    assert len(batches) == args.batch_size

    for i in range(len(batches)):
        batches[i].sort(key=lambda x: float(x[1]))

    with open(args.save, 'w') as fout:
        max_len = max(len(batch) for batch in batches)
        for i in range(max_len):
            for batch in batches:
                if len(batch) <= i:
                    continue
                fout.write(f'{int(batch[i][0])}\n')

