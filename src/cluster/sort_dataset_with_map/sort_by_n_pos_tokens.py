import argparse
import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the directory with input dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the directory where sorted dataset will be saved')
    parser.add_argument('--num-proc', type=int, required=True, help='Number of processes for multiprocessing')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets\
        .load_from_disk(args.input)\
        .map(lambda item: {'n_pos': sum(1 for c in item['input_ids'] if int(c) > int(0))}, num_proc=args.num_proc)\
        .sort(column='n_pos')
    dataset.remove_columns_('n_pos')
    dataset.save_to_disk(args.output)

