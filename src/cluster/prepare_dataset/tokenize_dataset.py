from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
import re
from transformers import BertTokenizer
from argparse import ArgumentParser

MIN_LENGTH = 25


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to the directory with dataset',
                        default='/home/aomelchenko/datasets/wiki40b_en_3M')
    parser.add_argument('--save', type=str,
                        help='Path to the EMPTY directory, where SPLITTED dataset will be saved',
                        default='/home/aomelchenko/datasets/wiki40b_en_3M_tokenized512')
    parser.add_argument('--tokenizer', type=str, help='Path the directory with tokenizer',
                        default='/home/aomelchenko/tokenizers/BertTokenizerBase')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = load_from_disk(args.dataset)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    dataset = dataset.map(lambda x: tokenizer(x['text'], max_length=512, truncation=True))
    dataset.save_to_disk(args.save)
