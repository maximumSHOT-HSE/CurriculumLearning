from datasets import Dataset, load_from_disk
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser

MIN_LENGTH = 25


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with dataset',
                        default='/home/aomelchenko/datasets/wiki40b_en_reduced')
    parser.add_argument('--save', type=str, help='Path to the EMPTY directory, where reduced dataset will be saved',
                        default='/home/aomelchenko/datasets/wiki40b_en_3M')
    return parser.parse_args()


def process_part(ds):
    sentences = []
    for record in tqdm(ds):
        if randint(0, 2) == 0:
            sentences.append(record['text'])

    return Dataset.from_dict({'text': sentences})


def process_dataset(ds):
    ds['validation'] = process_part(ds['validation'])
    ds['test'] = process_part(ds['test'])
    ds['train'] = process_part(ds['train'])


if __name__ == '__main__':
    args = parse_args()
    dataset = load_from_disk(args.dataset)
    process_dataset(dataset)
    dataset.save_to_disk(args.save)
