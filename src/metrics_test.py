import unittest
import sentencepiece as spm
import pickle
# import src.metrics as metrics
import metrics
from tqdm import tqdm


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


if __name__ == '__main__':
    unittest.main()
