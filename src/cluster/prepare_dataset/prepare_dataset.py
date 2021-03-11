from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
import re
from argparse import ArgumentParser

MIN_LENGTH = 25


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with dataset',
                        default='/home/aomelchenko/datasets/wiki40b_en_init')
    parser.add_argument('--save', type=str, help='Path to the EMPTY directory, where SPLITTED dataset will be saved',
                        default='/home/aomelchenko/datasets/wiki40b_en_splitted')
    return parser.parse_args()


def split(text, sentences):
    parts = re.split(r'([.;!?:\n]|_[A-Z_]+_)', text)
    for sentence in parts:
        if len(sentence) >= MIN_LENGTH:
            sentences.append(sentence.lower().strip())


def process_part(ds):
    sentences = []
    for record in tqdm(ds):
        split(record['text'], sentences)

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
