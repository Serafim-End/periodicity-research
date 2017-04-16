# coding: utf-8

from copy import deepcopy

import numpy as np

from sequence import PrepareSequence


class CONVAlgorithm(object):
    def __init__(self, sequence, threshold=0, verbose=False):
        self.sequence_o = PrepareSequence(sequence)
        self.sequence = self.sequence_o.bin_code()

        self.threshold = threshold
        self.candidates = {}
        self.verbose = verbose

    def _reverse(self):
        sigma = len(self.sequence_o.alphabet)
        return reduce(
            lambda r, e: r + e,
            [self.sequence[sigma * (i - 1):sigma * i]
             for i in range(len(self.sequence) / sigma, 0, -1)], []
        )

    def F2(self, p, l, candidates):
        sigma = len(self.sequence_o.alphabet)
        s = self.sequence_o.symbolic[np.ceil(candidates[0] / sigma)]
        for i in range(1, len(candidates)):
            if s != self.sequence_o.symbolic[np.ceil(candidates[i] / sigma)]:
                return 0

        len_symb = len(self.sequence_o.symbolic)
        PIpl = [self.sequence_o.symbolic[l + i * p] for i in range((len_symb - l) / p)]

        count = 0
        for e in PIpl:
            count = count + 1 if e == s else 0

        return count - 1

    def get_candidates(self):
        sigma = len(self.sequence_o.alphabet)
        n = len(self.sequence)
        c_hat = []

        if self.verbose:
            print(self.sequence_o.alphabet_symbol)
            print(self.sequence_o.alphabet)

        for i in range(0, len(self.sequence)):
            accumulator = 0

            if self.verbose:
                print('sequen:   ', self.sequence)
                print('subseq:', ' * '*i, self.sequence[:n - i])

            for j in range(n - i):
                accumulator += (2 ** j) * self.sequence[n - 1 - j] * self.sequence[n - 1 - i - j]

                if self.verbose:
                    print('j:', j,
                          ' n-1-{} = {}: {}'.format(
                              j, n-1-j, self.sequence[n - 1 - j]
                          ),
                          ' n-1-{}-{} = {}: {}'.format(
                              i, j, n-1-i-j, self.sequence[n - 1 - i - j]
                          ),
                          ' power: ', 2 ** j)

            c_hat.append(accumulator)

        c = [c_hat[i] for i in range(len(self.sequence)) if (i % sigma) == 0]

        if self.verbose:
            print('c_hat: ', c_hat)
            print('c: ', c)

        for p in range(1, (len(self.sequence) / (2 * sigma)) + 1):
            # reversed binary variant of cp = c[p]
            cp = map(int, bin(c[p])[2:])[::-1]
            wp = [i for i, e in enumerate(cp) if e == 1]

            if self.verbose:
                print('PERIOD SIZE: ', p)
                print('p: {}, wp: {}'.format(p, wp))

            for k in range(sigma):
                wpk = [wp[i] for i in range(len(wp)) if wp[i] % sigma == k]

                if self.verbose:
                    print('W{},{}'.format(p, k), wpk)

                for l in range(p - 1):
                    wpkl = [wpk[j] for j in range(len(wpk))
                            if ((n / sigma) - p - 1 - np.floor(wpk[j] / sigma)) % p == l]

                    if self.verbose:
                        print('W{},{},{}'.format(p,k,l), wpkl)

                    if wpkl and (self.F2(p, l, wpkl) / np.ceil(float(n - l) / p)) >= self.threshold:

                        if p not in self.candidates:
                            self.candidates[p] = []

                        self.candidates[p].append([
                            self.sequence_o.alphabet[k], l, wpkl
                        ])

        return self.candidates

    def get_partial(self):

        self.get_candidates()

        n = len(self.sequence)
        sigma = len(self.sequence_o.alphabet)

        def _new_wp(wp, new_wpkl):
            if len(new_wpkl) < len(wp):
                return wp

            if len(new_wpkl) > len(wp):

                if len(wp) == 0:
                    for el in new_wpkl:
                        wp.append([el])
                    return wp

                for i in range(len(wp)):
                    wp[i].append(new_wpkl[i])

                for i in range(len(new_wpkl) - len(wp)):
                    wp.append([])

                    if len(wp) > 1:
                        for i in range(len(wp[0]) - 1):
                            wp[-1].append(0)

                    wp[-1].append(new_wpkl[i])

            if len(new_wpkl) == len(wp):
                for i in range(len(wp)):
                    wp[i].append(new_wpkl[i])

            return wp

        self.partial_candidates = []
        for p, c in self.candidates.iteritems():
            # we can concatenate more than one element
            if len(c) < 2:
                continue

            wp, pattern, bypasses = [], [], []
            for e in c:
                sk, l, wpkl, = e

                if sk in pattern or l in bypasses:
                    continue

                old_wp = deepcopy(wp)
                new_wp = _new_wp(wp, wpkl)

                flag = False
                for ar in new_wp:
                    f_val = np.ceil(((n / sigma) - p - 1 - np.ceil(ar[0] / sigma)) / p)
                    for i in range(1, len(ar)):
                        n_val = np.ceil(((n / sigma) - p - 1 - np.ceil(ar[i] / sigma)) / p)
                        if n_val == f_val:
                            f_val = n_val
                        else:
                            new_wp = wp
                            flag = True
                            break
                    if flag:
                        break

                if old_wp != new_wp:
                    pattern.append(sk)
                    bypasses.append(l)

                wp = new_wp
                self.partial_candidates.append(
                    [p,
                     list(pattern),
                     list(bypasses),
                     wp,
                     float(len(wp))/np.ceil(n / (sigma * p))]
                )

        return self.partial_candidates

    def get_best_candidates(self):
        self.get_partial()

        if self.verbose:
            print('symbol candidates: {}'.format(self.candidates))
            print('partial segment: {}'.format(self.partial_candidates))

        return np.unique(self.candidates.keys() +
                         [ar[0] for ar in self.partial_candidates])


class CONVSegmentAlgorithm(object):
    """
    mapping scheme should follow the rule:
    F(e_(i)) * F(e_(i-j)) != 0 if e_(i) == e_(i-j) else 0
    """

    def __init__(self, sequence, threshold):
        self.sequence_o = PrepareSequence(sequence)
        self.sequence = self.sequence_o.bin_code()

        self.periodicity_threshold = threshold
        self.candidates = []

    def get_candidates(self):
        # Use partial in CONVAlgorithm as more general approach
        pass
