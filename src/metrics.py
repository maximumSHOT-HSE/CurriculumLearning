import numpy as np


def excess_entropy_slow(text: str, single_H, pair_H):
    """
    Calculates excess entropy of given string in O(n^2) time complexity in a straightforward way

    :param text: an input tokenized string
    :param single_H: a function that calculates H(i, x_i)
    :param pair_H: a function that calculates H(x_i | x_{i-1}) = H(i, x_i, x_{i-1})
    :return: a float value which is equal to excess entropy of given input string
    """

    n = len(text)

    def calculate_excess_entropy_on_range(l: int, r: int) -> float:  # [l, r)
        H = 0
        for i in range(l, r):
            if i == l:
                H += single_H(i, text[i])
            else:  # i > l
                H += pair_H(i, text[i], text[i - 1])
        return H

    EE = 0
    for i in range(1, n):
        EE += calculate_excess_entropy_on_range(0, i)
        EE += calculate_excess_entropy_on_range(i, n)

    return EE - (n - 1) * calculate_excess_entropy_on_range(0, n)


def excess_entropy_fast(text: str, single_H, pair_H):
    """
    Calculates excess entropy of given string in O(n) time complexity

    :param text: an input tokenized string
    :param single_H: a function that calculates H(i, x_i)
    :param pair_H: a function that calculates H(x_i | x_{i-1}) = H(i, x_i, x_{i-1})
    :return: a float value which is equal to excess entropy of given input string
    """

    n = len(text)
    EE = 0
    for i in range(n - 1):
        EE += single_H(i + 1, text[i + 1]) - pair_H(i + 1, text[i + 1], text[i])
    return EE


if __name__ == '__main__':
    print('Hello, world')
