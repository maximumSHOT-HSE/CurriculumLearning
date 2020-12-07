import sentencepiece as spm
import numpy as np
import nltk
from tqdm import tqdm
import hashlib
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spm', type=str, help='Path to the file with spm tokenizer')
    parser.add_argument('--corpus', type=str, help='Path to the corpus file which should be processes')
    parser.add_argument('--save-prefix', type=str, help='Prefix of file to be saved after corpus processing')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    F_pos = nltk.FreqDist()
    F_single = nltk.FreqDist()
    F_pair = nltk.FreqDist()
    F_last = nltk.FreqDist()

    with open(args.corpus, 'r') as file:
        for line in tqdm(file):
            line = line.strip()
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

    with \
        open(f'{args.save_prefix}_pos', 'wb') as out_pos, \
        open(f'{args.save_prefix}_single', 'wb') as out_single, \
        open(f'{args.save_prefix}_pair', 'wb') as out_pair, \
        open(f'{args.save_prefix}_last', 'wb') as out_last:

        pickle.dump(F_pos, out_pos)
        pickle.dump(F_single, out_single)
        pickle.dump(F_pair, out_pair)
        pickle.dump(F_last, out_last)
