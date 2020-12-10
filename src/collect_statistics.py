import argparse
import pickle

import nltk
import sentencepiece as spm
import tensorflow_datasets as tfds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spm', type=str, help='Path to the file with spm tokenizer')
    parser.add_argument('--corpus', type=str, help='Path to the corpus file which should be processes '
                                                   'or name of tfds')
    parser.add_argument('--save-prefix', type=str, help='Prefix of file to be saved after corpus processing')
    parser.add_argument('--mode', type=str, choices=['file', 'tfds'], required=True)
    parser.add_argument('--dump', type=str, help='Path to the dump directory, where tfds is stored')
    parser.add_argument('--logging-period', type=int, default=1000)
    return parser.parse_args()


def get_from_file_generator(path: str):
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            yield line


def get_tfds_generator(args):
    dataset = tfds.load(
        name=args.corpus,
        data_dir=args.dump,
        with_info=False,
        split=tfds.Split.TRAIN,
        shuffle_files=False,
        download=False
    )
    for item in dataset:
        yield str(item['text'].numpy().decode('utf-8')).strip()


if __name__ == '__main__':
    args = parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    F_pos = nltk.FreqDist()
    F_single = nltk.FreqDist()
    F_pair = nltk.FreqDist()
    F_last = nltk.FreqDist()

    generator = {
        'file': get_from_file_generator(args.corpus),
        'tfds': get_tfds_generator(args)
    }[args.mode]

    for i, line in enumerate(generator):
        x = sp.encode_as_ids(line)
        n = len(x)
        if n == 0:
            continue
        bgs = nltk.bigrams(x)

        add_pos = nltk.FreqDist(range(n))  # i
        add_single = nltk.FreqDist(zip(range(n), x))  # (i, x[i])
        add_pair = nltk.FreqDist(zip(range(1, n), bgs))  # (i, (x_prv, x_cur))

        F_pos += add_pos
        F_single += add_single
        F_pair += add_pair
        F_last[(n - 1, x[n - 1])] += 1

        if i % args.logging_period == 0:
            print('\n==========\n')
            print(f'{i} iterations, |F_pos| = {len(F_pos)}, |F_single| = {len(F_single)}, '
                  f'|F_pair| = {len(F_pair)}, |F_last| = {len(F_last)}')
            print(f'sample = {line}')

    with \
        open(f'{args.save_prefix}_pos', 'wb') as out_pos, \
        open(f'{args.save_prefix}_single', 'wb') as out_single, \
        open(f'{args.save_prefix}_pair', 'wb') as out_pair, \
        open(f'{args.save_prefix}_last', 'wb') as out_last:

        pickle.dump(F_pos, out_pos)
        pickle.dump(F_single, out_single)
        pickle.dump(F_pair, out_pair)
        pickle.dump(F_last, out_last)
