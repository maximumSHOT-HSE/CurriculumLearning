import numpy as np
import metrics
import pickle
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    text_path = '../data/corpus_small.txt'
    sp = spm.SentencePieceProcessor()
    sp.load('../data/sew_bpe.model')
    with \
        open('../data/sew_stat_small_pos', 'rb') as f_pos_in, \
        open('../data/sew_stat_small_single', 'rb') as f_single_in, \
        open('../data/sew_stat_small_pair', 'rb') as f_pair_in, \
        open('../data/sew_stat_small_last', 'rb') as f_last_in:
        F_pos = pickle.load(f_pos_in)
        F_single = pickle.load(f_single_in)
        F_pair = pickle.load(f_pair_in)
        F_last = pickle.load(f_last_in)

    lens = []
    ees = []

    MAX_LEN = 200

    with open(text_path, 'r') as text_fin:
        for line in tqdm(text_fin):
            line = line.strip()
            x = sp.encode_as_ids(line)
            fast_ee = metrics.excess_entropy_fast(
                x,
                metrics.get_H_single(F_pos, F_single),
                metrics.get_H_pair(F_pos, F_single, F_pair, F_last)
            )
            if len(x) < MAX_LEN:
                lens.append(len(x))
                ees.append(fast_ee)

    mean_ee = np.zeros(MAX_LEN, dtype=float)
    cnt = np.zeros(MAX_LEN, dtype=float)
    median = np.zeros(MAX_LEN, dtype=float)
    ar = [[] for _ in range(MAX_LEN)]

    for len_i, ee_i in zip(lens, ees):
        mean_ee[len_i] += ee_i
        cnt[len_i] += 1
        ar[len_i].append(ee_i)

    for i in range(MAX_LEN):
        n = len(ar[i])
        if n == 0:
            continue
        ar[i].sort()
        median[i] = ar[i][n // 2]

    mean_ee /= (cnt + 1e-10)
    mask = cnt > 0

    plt.figure(figsize=(20, 20))
    plt.title('10k texts from Simple English Wikipedia')
    plt.xlabel('length')
    plt.ylabel('Excess Entropy')
    plt.scatter(lens, ees)
    plt.scatter(np.arange(MAX_LEN)[mask], mean_ee[mask], label='mean ee')
    # plt.scatter(np.arange(MAX_LEN)[mask], median[mask], label='median ee')
    plt.legend()
    plt.show()
