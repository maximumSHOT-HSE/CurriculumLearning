import argparse
import datasets
from transformers import BertTokenizer
from transformers import EvaluationStrategy
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from copy import deepcopy

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--max-noise-level', type=float, required=True)
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


def add_keyboard_aug(x, max_noise_level):
    noise_level = np.random.rand() * max_noise_level
    aug = nac.KeyboardAug(aug_word_p=noise_level)
    return {'text': aug.augment(x['text']), 'noise_level': noise_level}


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    print(dataset)
    print()

    for i, x in enumerate(dataset['train']):
        print(x)
        if i >= 10:
            break

    dataset['train'] = dataset['train'].map(lambda x: add_keyboard_aug(x, args.max_noise_level))

    print(dataset)
    print()

    for i, x in enumerate(dataset['train']):
        print(x)
        if i >= 10:
            break

    dataset.save_to_disk(args.save)
