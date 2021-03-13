import numpy as np

def test(K, S):
    W = np.zeros(K)

    for q in range(K + 2 * S - 2):
        t = q - S + 1
        w = np.zeros(K)
        for i in range(S + 1):
            for id in [t - S + i, t + S - i]:
                if 0 <= id < K:
                    w[id] = i / (S ** 2)
        assert w.sum() > 0
        W += w

    print()
    print(W)

test(K=40, S=6)

assert 1 == 2, "1 != 2"
