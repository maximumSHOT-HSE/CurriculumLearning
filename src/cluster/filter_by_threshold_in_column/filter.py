import argparse
import datasets
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the directory with input dataset')
    parser.add_argument('--column', type=str, required=True, help='Column name to sort by')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold value. Only rows, s.t. x[column] < threshold will remain')
    parser.add_argument('--output', type=str, required=True, help='Path to the directory where sorted dataset will be saved')
    return parser.parse_args()


def filter_indices(ds, column, threshold):
    indices = []
    for i, x in tqdm(enumerate(ds)):
        if x[column] < threshold:
            indices.append(i)
    return indices


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.input)
    dataset['train'] = dataset['train'].select(indices=filter_indices(dataset['train'], args.column, args.threshold))
    dataset.save_to_disk(args.output)
