import argparse
import datasets
from transformers import BertTokenizer
from bs4 import BeautifulSoup
import pandas as pd
import random
import math
from custom_trainers import FromFileSampler


if __name__ == '__main__':
    sampler = FromFileSampler([], '/home/aomelchenko/Bachelor-s-Degree/src/cluster/fine_tuning/sentiment140/s140_sort_by_len_merge_64.txt')
    xs = list(sampler)
    print(f'len = {len(xs)}')
    for i, x in enumerate(xs):
        print(i, x)
        if i >= 15:
            break

