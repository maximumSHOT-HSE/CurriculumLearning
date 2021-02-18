import argparse
import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the directory with input dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.input)

    for part in dataset:
        print()
        print('part', part)
        for i, x in enumerate(dataset[part]):
            print(len(x['input_ids']))
            if i >= 50:
                break

