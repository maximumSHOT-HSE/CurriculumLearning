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
        .map(lambda item: {'tse_div_len': item['tse'] / (len(item['input_ids']) + 1)}, num_proc=args.num_proc)\
        .sort(column='tse_div_len')
    dataset.remove_columns_('tse_div_len')
    dataset.save_to_disk(args.output)
