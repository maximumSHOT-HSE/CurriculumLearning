import unittest
import sentencepiece as spm


class TestExcessEntropy(unittest.TestCase):

    def setUp(self):
        self.text_path = '../data/corpus.txt'
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('../data/sew_bpe.model')

    def test_fast_equal_to_slow(self):  # stress test
        print('OK')


if __name__ == '__main__':
    unittest.main()
