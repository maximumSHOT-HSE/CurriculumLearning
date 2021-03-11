import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import pandas as pd
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--repeat', type=bool, default=False)
    parser.add_argument('--builder', type=str, required=True, choices=['lr', 'mlr'])
    return parser.parse_args()


def group_into_batches(xs: list, batch_size: int):
    batches = []
    n = len(xs)
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        # [i, j)
        batches.append(xs[i: j])
    return batches

def calc_median_in_batch(batch: list):
    n = len(batch)
    if n == 0:
        return 0
    batch = list(batch)
    batch.sort(key=lambda x: float(x[1]))
    return float(batch[n // 2][1])


if __name__ == '__main__':
    args = parse_args()

    print(args.dump_file)

    with open(args.dump_file, 'r') as fin:
        dataset = []
        for line in fin:
            dataset.append(line.strip('\x00').strip().split(' '))

    # i, tse, len

    dataset = np.array(dataset)

    print(f'dataset shape = {dataset.shape}')
    print(f'repeat = {args.repeat}')

    if args.repeat:
        indices = np.random.randint(low=0, high=len(dataset), size=len(dataset))
        dataset = dataset[indices]
    else:
        random.shuffle(dataset)

    dataset = list(dataset)

    batches = group_into_batches(dataset, args.batch_size)

    sum_len = 0
    for batch in batches:
        sum_len += len(batch)

    batches.sort(key=calc_median_in_batch)

    if args.builder == 'mlr':  # from mid to borders
        n = len(batches)
        mid = n // 2
        le = mid
        ri = mid
        new_batches = []
        for i in range(n):
            if (i % 2 == 0 and ri < n) or (i % 2 == 1 and le <= 0):
                new_batches.append(batches[ri])
                ri += 1
            else:
                le -= 1
                new_batches.append(batches[le])
        batches = new_batches

    print(f'sum_len = {sum_len}')
    print(f'total len = {len(dataset)}')
    print(f'batches count = {len(batches)}')

    with open(args.save, 'w') as fout:
        for batch in batches:
            random.shuffle(batch)
            for x in batch:
                fout.write(f'{x[0]}\n')

