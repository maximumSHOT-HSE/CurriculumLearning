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
            # print(x['labels'], sum(1 for c in x['input_ids'] if int(c) > int(0)))
            # print(x['excess_entropy'], x['text'])
            # print(f'{len(x["text"])}, {x["tse"]}')
            print(x)
            if i >= 50:
                break

