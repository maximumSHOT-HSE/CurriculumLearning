import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import datasets
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with dataset', required=True)
    parser.add_argument('--config', type=str, help='Path to the file with statistics config', required=True)
    parser.add_argument('--save', type=str, help='Path to the EMPTY directory, where statistics will be saved', required=True)
    parser.add_argument('--start', type=int, help='Start item id (inclusive)', required=True)
    parser.add_argument('--end', type=int, help='End item id (exclusive)', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset)
    with open(args.config) as config_fp:
        config = json.load(config_fp)

    config['start'] = args.start
    config['end'] = args.end
    config['dataset'] = args.dataset

    BASE = config['hash-base']
    BASE_SQR = BASE ** 2

    len_counter = Counter()
    last_counter = Counter()
    single_counter = Counter()
    pair_counter = Counter()

    mask = np.arange(args.start, args.end)

    for it, item in tqdm(enumerate(dataset.select(mask))):
        x = np.array(item['input_ids'])
        n = x.shape[0]
        i = np.arange(n)

        len_counter[n] += 1
        last_counter[ (n - 1) + x[-1] * BASE ] += 1

        single_h = i + x * BASE
        pair_h = i[:-1] + x[:-1] * BASE + x[1:] * BASE_SQR

        single_counter += Counter(single_h)
        pair_counter += Counter(pair_h)

    root = Path(args.save)

    paths = {
        root / 'len_counter.bin': len_counter,
        root / 'last_counter.bin': last_counter,
        root / 'single_counter.bin': single_counter,
        root / 'pair_counter.bin': pair_counter
    }

    for path, counter in paths.items():
        with open(path, 'wb') as f:
            pickle.dump(counter, f)

    with open(root / 'config.json', 'w') as f:
        json.dump(config, f)
