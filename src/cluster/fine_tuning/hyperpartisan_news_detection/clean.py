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

    print(f"len = {len(dataset['train'])}")

    print('===================================================')
    for part in ['train', 'validation']:
        print('PART', part)
        for i, x in enumerate(dataset[part]):
            raw = x['text']
            print(raw)
            # ids = tokenizer(x['text'])
            # print(ids)
            # print(len(ids['input_ids']))
            print('-----------')
            print(BeautifulSoup(raw, 'lxml').text)
            print('NEXTNEXTNEXT')
            if i >= 5:
                break
