import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    print('===================================================')
    print(dataset)
    print('train INFO')
    print(dataset['train'].info)
    print('test INFO')
    print(dataset['test'].info)

    print(f"len = {len(dataset['train'])}")

    print('===================================================')
    for part in ['train', 'test']:
        print('PART', part)
        for i, x in enumerate(dataset[part]):
            print(sum(1 for c in x['text'] if c == ' '), len(x['text']), x)
            # print(len(x['text']), len(x['input_ids']), x['label'], x['text'], tokenizer.convert_ids_to_tokens(x['input_ids']))
            # ids = tokenizer(x['text'])
            # print(ids)
            # print(len(ids['input_ids']))
            if i >= 500:
                break

    print()
    print(tokenizer.vocab)

