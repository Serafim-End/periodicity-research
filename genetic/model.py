# coding: utf-8

import numpy as np
import settings

# For testing was used http://victoria.biengi.ac.ru/splinter/index.php
# KOROTKOV implementation


class Organism(object):
    def __init__(self, n, S, S1, alphabet_size):
        self.n = n
        self.S = S
        self.S1 = S1
        self.alphabet_size = alphabet_size
        self.Fmax = 0

        self.m = self.generate_weight_matrix()

    def fitness(self):
        pass

    def div_m_calculation(self):
        r2 = np.sum(self.m * self.m)

        kd = 0
        tj = 1. / self.n
        p = np.zeros(self.m.shape[0], self.m.shape[1])

        for i in range(self.m.shape[0]):
            fi = np.count_nonzero(self.S == i).astype(float) / self.S.size
            for j in range(self.m.shape[1]):
                p[i, j] = fi * tj
                kd += self.m[i, j] * p[i, j]

    def dynamic(self):

        self.F = np.zeros((self.S.size, self.S.size))
        self.F1 = np.zeros((self.S.size, self.S.size))
        self.F2 = np.zeros((self.S.size, self.S.size))
        self.dm = None
        d = None
        e = d / 4.

        for i in range(1, self.S.size):
            for j in range(1, self.S.size):
                self.F[i, j] = np.max([
                    0,
                    self.F[i - 1, j - 1] + self.dm[self.S[i], self.S1[j]],
                    self.F1[i - 1, j - 1] - d,
                    self.F2[i - 1, j - 1] - d
                ])

                self.F1[i, j] = np.max([
                    self.F[i - 1, j] - d,
                    self.F1[i - 1, j] - e
                ])

                self.F2[i, j] = np.max([
                    self.F[i, j - 1] - d,
                    self.F2[i, j - 1] - e
                ])

    def __str__(self):
        pass

    def __cmp__(self, other):
        return np.sqrt(np.sum(other - self.m) ** 2) > settings.D0

    def cmp_all(self, others):
        return all([self > o for o in others])

    def generate_Sri(self, S, alphabet_size):
        Sri = []
        for i in range(S.size):
            r = np.random.random()
            Sri.append(
                int(r * alphabet_size) if r > 0.5 else S[i]
            )

        return np.array(Sri)

    def generate_weight_matrix(self):
        row, column = settings.SIZE_PERIOD_TYPES, self.n
        V = np.zeros((row, column))

        Sri = self.generate_Sri(self.S, self.alphabet_size)
        v = np.array(
            map(lambda srk, s1k: 1 if srk == s1k else 0, Sri, self.S1)
        )

        curr = 0
        for i in range(row):
            for j in range(column):
                if len(v) > curr:
                    V[i, j] = v[curr]
                curr += 1

        x, y, N = V.sum(axis=0), V.sum(axis=1), V.sum()
        m = np.zeros((row, column))
        for i in range(column):
            for j in range(row):
                pij = float(x[i] * y[j]) / N
                div = np.sqrt(pij * (1 - (float(pij) / N)))
                m[j, i] = (V[j, i] - pij) / div if div != 0 else 0
        return m


class Population(object):
    def __init__(self):
        self.organisms = []
        self.size = len(self.organisms)

    def get_max_Fmax(self):
        curr_Fmax = -np.inf
        for o in self.organisms:
            if o.Fmax > curr_Fmax:
                curr_Fmax = o.Fmax
        return curr_Fmax

    def sort(self):
        self.organisms = self.organisms.sort(key=lambda e: e.Fmax)
        return self.organisms

    def __str__(self):
        pass


