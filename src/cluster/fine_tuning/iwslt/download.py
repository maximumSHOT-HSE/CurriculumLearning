import argparse
import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_dataset('ted_talks_iwslt', 'nl_en_2014')
    print(dataset)
    dataset.save_to_disk(args.save)

