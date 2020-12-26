import argparse
import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the directory with input dataset')
    parser.add_argument('--column', type=str, required=True, help='Column name to sort by')
    parser.add_argument('--output', type=str, required=True, help='Path to the directory where sorted dataset will be saved')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.input).sort(column=args.column).save_to_disk(args.output)
