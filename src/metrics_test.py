import unittest
import sentencepiece as spm
import pickle
import metrics
from tqdm import tqdm
import nltk
import numpy as np


class TestExcessEntropy(unittest.TestCase):

    def setUp(self):
        self.text_path = '../data/test_corpus.txt'
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('../data/sew_bpe.model')
        with \
            open('../data/sew_stat_small_pos', 'rb') as f_pos_in, \
            open('../data/sew_stat_small_single', 'rb') as f_single_in, \
            open('../data/sew_stat_small_pair', 'rb') as f_pair_in, \
            open('../data/sew_stat_small_last', 'rb') as f_last_in:
            self.F_pos = pickle.load(f_pos_in)
            self.F_single = pickle.load(f_single_in)
            self.F_pair = pickle.load(f_pair_in)
            self.F_last = pickle.load(f_last_in)
        self.eps = 1e-10

    def test_fast_equal_to_slow(self):  # stress test
        with open(self.text_path, 'r') as text_fin:
            for line in text_fin:
                line = line.strip()
                x = self.sp.encode_as_ids(line)
                slow_ee = metrics.excess_entropy_slow(
                    x,
                    metrics.get_H_single(self.F_pos, self.F_single),
                    metrics.get_H_pair(self.F_pos, self.F_single, self.F_pair, self.F_last)
                )
                fast_ee = metrics.excess_entropy_fast(
                    x,
                    metrics.get_H_single(self.F_pos, self.F_single),
                    metrics.get_H_pair(self.F_pos, self.F_single, self.F_pair, self.F_last)
                )
                abs_error = abs(slow_ee - fast_ee)
                self.assertTrue(abs_error < self.eps)

    def test_simple(self):
        data = [
            [2, 1, 1, 0, 2, 2, 2, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 2],
            [2, 0, 1, 2, 1, 1, 2, 0, 1],
            [2, 0, 0, 0, 0, 1, 0, 0],
            [1, 2, 1, 0, 2, 0, 0],
            [2, 0, 0, 2, 2, 2],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 2, 2, 2],
            [1, 1],
            [2, 1],
        ]
        text = [1, 1, 2, 1, 0, 2, 2, 0, 1]

        h_single = [
            metrics.calculate_entropy([3.0 / 10.0, 7.0 / 10.0]),
            metrics.calculate_entropy([3.0 / 10.0, 7.0 / 10.0]),
            metrics.calculate_entropy([1.0 / 8.0, 7.0 / 8.0]),
            metrics.calculate_entropy([1.0 / 8.0, 7.0 / 8.0]),
            metrics.calculate_entropy([3.0 / 8.0, 5.0 / 8.0]),
            metrics.calculate_entropy([2.0 / 7.0, 5.0 / 7.0]),
            metrics.calculate_entropy([2.0 / 5.0, 3.0 / 5.0]),
            metrics.calculate_entropy([2.0 / 4.0, 2.0 / 4.0]),
            metrics.calculate_entropy([1.0 / 3.0, 2.0 / 3.0]),
        ]

        h_pair_joint = [
            0,
            metrics.calculate_entropy(np.array([1, 2, 2, 5], dtype=float) / 10.0),
            metrics.calculate_entropy(np.array([0, 1, 1, 6], dtype=float) / 8.0),
            metrics.calculate_entropy(np.array([0, 1, 1, 6], dtype=float) / 8.0),
            metrics.calculate_entropy(np.array([1, 2, 0, 5], dtype=float) / 8.0),
            metrics.calculate_entropy(np.array([0, 2, 3, 2], dtype=float) / 7.0),
            metrics.calculate_entropy(np.array([1, 1, 0, 3], dtype=float) / 5.0),
            metrics.calculate_entropy(np.array([1, 1, 1, 1], dtype=float) / 4.0),
            metrics.calculate_entropy(np.array([1, 0, 0, 2], dtype=float) / 3.0)
        ]

        h_pair = [0 for _ in range(9)]
        for i in range(1, 9):
            h_pair[i] = h_pair_joint[i] - h_single[i - 1]

        custom_ee = 0
        for i in range(9):
            custom_ee += (h_single[i] if i == 0 else h_pair[i])
        custom_ee *= -(9 - 1)
        for i in range(9):
            for j in range(i):
                custom_ee += (h_single[j] if j == 0 else h_pair[j])
            for j in range(i + 1, 9):
                custom_ee += (h_single[j] if j == i + 1 else h_pair[j])

        f_pos = nltk.FreqDist()
        f_single = nltk.FreqDist()
        f_pair = nltk.FreqDist()
        f_last = nltk.FreqDist()

        for sample in data:
            n = len(sample)
            for i in range(n):
                f_pos[i] += 1
                f_single[(i, sample[i])] += 1
                if i > 0:
                    f_pair[(i, (sample[i - 1], sample[i]))] += 1
            f_last[(n - 1, sample[n - 1])] += 1

        fast_ee = metrics.excess_entropy_fast(
            text,
            metrics.get_H_single(f_pos, f_single),
            metrics.get_H_pair(f_pos, f_single, f_pair, f_last)
        )

        abs_error = abs(custom_ee - fast_ee)
        self.assertTrue(abs_error < self.eps)


if __name__ == '__main__':
    unittest.main()
