import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import re
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--min-length', type=int, required=True)
    return parser.parse_args()


def get_count_of_char(text, ch):
    return sum(1 for c in text if c == ch)


def split_text_into_chunks(text, min_length, max_length):
    parts = re.split(r'([\.;!?:\n]|\.{3})', text)
    texts = []
    n = len(parts)
    i = 0
    while i < n:
        if get_count_of_char(parts[i], ' ') > max_length:
            texts.append(parts[i])
            i += 1
            continue
        j = i
        current_text = ''
        while j < n and get_count_of_char(current_text + parts[j], ' ') <= max_length:
            current_text += parts[j]
            j += 1
        if get_count_of_char(current_text, ' ') >= min_length:
            texts.append(current_text)
        i = j
    print(f'n_texts = {len(texts)} : ')
    print(f'init text len = {len(text)}')
    return texts


def split_texts_in_dataset(dataset, min_length, max_length):
    texts = []
    labels = []
    for i, x in tqdm(enumerate(dataset)):
        current_texts = split_text_into_chunks(x['text'], min_length, max_length)
        current_labels = [x['hyperpartisan'] for _ in range(len(current_texts))]
        texts += current_texts
        labels += current_labels
    return datasets.Dataset.from_dict({'text': texts, 'hyperpartisan': labels})


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    for part in dataset:
        dataset[part] = split_texts_in_dataset(dataset[part], args.min_length, args.max_length)

    print(dataset)

    dataset.save_to_disk(args.save)

