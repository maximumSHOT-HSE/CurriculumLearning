import argparse
import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_dataset('hyperpartisan_news_detection', 'bypublisher')
    print(dataset)
    dataset.save_to_disk(args.save)

