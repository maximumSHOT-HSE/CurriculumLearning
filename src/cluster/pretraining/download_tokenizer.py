import argparse
from transformers import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = BertTokenizer.from_pretrained(args.model_name)
    print(model)
    model.save_pretrained(args.save)

