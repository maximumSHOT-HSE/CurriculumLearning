from typing import Union

import nltk
import numpy as np


def calculate_entropy(distribution: Union[list, np.ndarray]) -> float:
    EPS = 1e-9
    p = np.array(distribution)
    return (-p * np.log(p + EPS) - (1 - p) * np.log(1 - p + EPS)).sum()


def H_single(i, x_i, F_pos: nltk.FreqDist, F_single: nltk.FreqDist):
    p = F_single[(i, x_i)] / F_pos[i]
    return calculate_entropy([p, 1 - p])


def get_H_single(F_pos: nltk.FreqDist, F_single: nltk.FreqDist):
    return lambda i, x_i: H_single(i, x_i, F_pos, F_single)


def get_H_pair(F_pos: nltk.FreqDist, F_single: nltk.FreqDist, F_pair: nltk.FreqDist, F_last: nltk.FreqDist):
    def H_pair(i, x_prv, x_cur):
        T = F_pos[i]
        c11 = F_pair[(i, (x_prv, x_cur))]
        c01 = F_single[(i, x_cur)] - c11
        c10 = F_single[(i - 1, x_prv)] - c11 - F_last[(i - 1, x_prv)]
        c00 = T - c11 - c01 - c10
        assert c00 >= 0
        return calculate_entropy(np.array([c00, c01, c10, c11], dtype=float) / T) - \
               H_single(i - 1, x_prv, F_pos, F_single)
    return H_pair


def excess_entropy_slow(text: str, H_single, H_pair):
    """
    Calculates excess entropy of given string in O(n^2) time complexity in a straightforward way

    :param text: an input tokenized string
    :param H_single: a function that calculates H(i, x_i)
    :param H_pair: a function that calculates H(x_i | x_{i-1}) = H(i, x_{i-1}, x_i)
    :return: a float value which is equal to excess entropy of given input string
    """

    n = len(text)

    def calculate_excess_entropy_on_range(l: int, r: int) -> float:  # [l, r)
        H = 0
        for i in range(l, r):
            if i == l:
                H += H_single(i, text[i])
            else:  # i > l
                H += H_pair(i, text[i - 1], text[i])
        return H

    EE = 0
    for i in range(1, n):
        EE += calculate_excess_entropy_on_range(0, i)
        EE += calculate_excess_entropy_on_range(i, n)

    return EE - (n - 1) * calculate_excess_entropy_on_range(0, n)


def excess_entropy_fast(text: str, H_single, H_pair):
    """
    Calculates excess entropy of given string in O(n) time complexity

    :param text: an input tokenized string
    :param H_single: a function that calculates H(i, x_i)
    :param H_pair: a function that calculates H(x_i | x_{i-1}) = H(i, x_{i-1}, x_i)
    :return: a float value which is equal to excess entropy of given input string
    """
    n = len(text)
    EE = 0
    for i in range(n - 1):
        tmp = H_single(i + 1, text[i + 1]) - H_pair(i + 1, text[i], text[i + 1])
        if tmp < 0:
            print(i, text[i], text[i + 1])
        EE += tmp
    return EE


def calculate_subset_mean_H_slow(text: str, H_single, H_pair) -> np.ndarray:
    """
    Calculates subset mean entropy for each 1 <= k <= n of given string in O*(2^n) time complexity

    :param text: an input tokenized string
    :param H_single: a function that calculates H(i, x_i)
    :param H_pair: a function that calculates H(x_i | x_{i-1}) = H(i, x_{i-1}, x_i)
    :return: (n,) np.ndarray with calculated subset mean entropy for each 1 <= k <= n (result[k - 1] = mean for k)
    """
    n = len(text)
    sumH = np.zeros(n)
    cnt = np.zeros(n)
    N = 2 ** n
    for mask in range(1, N):
        cnt_bits = 0
        prev_i = -10
        H = 0
        for i in range(n):
            if (2 ** i) & mask:  # i in mask
                cnt_bits += 1
                if prev_i + 1 == i:
                    H += H_pair(i, text[i - 1], text[i])
                else:
                    H += H_single(i, text[i])
                prev_i = i
        sumH[cnt_bits - 1] += H
        cnt[cnt_bits - 1] += 1
    return sumH / cnt


def calculate_subset_mean_H_fast(text: str, H_single, H_pair) -> np.ndarray:
    """
    Calculates subset mean entropy for each 1 <= k <= n of given string in O(n^2) time complexity

    :param text: an input tokenized string
    :param H_single: a function that calculates H(i, x_i)
    :param H_pair: a function that calculates H(x_i | x_{i-1}) = H(i, x_{i-1}, x_i)
    :return: (n,) np.ndarray with calculated subset mean entropy for each 1 <= k <= n (result[k - 1] = mean for k)
    """
    n = len(text)
    if n == 1:
        return np.array([H_single(0, text[0])])
    sum_h_pair = 0
    sum_h_single = 0
    for i in range(1, n):
        sum_h_pair += H_pair(i, text[i - 1], text[i])
        sum_h_single += H_single(i, text[i])
    ks = np.arange(n) + 1
    return ks * (ks - 1) * sum_h_pair / n / (n - 1) + \
           ks * (n - ks) * sum_h_single / n / (n - 1) + \
           H_single(0, text[0]) * ks / n


def TSE_slow(text: str, H_single, H_pair):
    """
    Calculates TSE of given string in O*(2^n) time complexity

    :param text: an input tokenized string
    :param H_single: a function that calculates H(i, x_i)
    :param H_pair: a function that calculates H(x_i | x_{i-1}) = H(i, x_{i-1}, x_i)
    :return: a float value which is equal to TSE of given input string
    """
    if len(text) == 0:
        return 0
    subset_mean_H = calculate_subset_mean_H_slow(text, H_single, H_pair)
    TSE = 0
    n = len(text)
    for k in range(1, n):
        C_k = (n / k) * subset_mean_H[k - 1] - subset_mean_H[n - 1]
        TSE += (k / n) * C_k
    return TSE


def TSE_fast(text: str, H_single, H_pair):
    """
    Calculates TSE of given string in O(n^2) time complexity

    :param text: an input tokenized string
    :param H_single: a function that calculates H(i, x_i)
    :param H_pair: a function that calculates H(x_i | x_{i-1}) = H(i, x_{i-1}, x_i)
    :return: a float value which is equal to TSE of given input string
    """
    if len(text) == 0:
        return 0
    subset_mean_H = calculate_subset_mean_H_fast(text, H_single, H_pair)
    return subset_mean_H[:-1].sum() - subset_mean_H[-1] * (len(text) - 1) / 2
