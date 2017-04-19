# coding: utf-8

import numpy as np


class GenerateData(object):

    def __init__(self, sequence_size, alphabet_size, period_size,
                 noise_level, sequences_count, path=None):

        self.sequence_size = sequence_size
        self.alphabet_size = alphabet_size
        self.period_size = period_size
        self.noise_level = noise_level
        self.sequences_count = sequences_count
        self.path = path

        self.sequences = []

    def generate(self):

        periods_count = self.sequence_size / self.period_size
        tail = self.sequence_size - periods_count * self.period_size

        for i in range(self.sequences_count):

            period = np.random.choice(
                a=range(self.alphabet_size),
                size=self.period_size,
                replace=False
            )

            # the theory
            # with some probability certain period will be changed
            # may be just one elelent will be changed
            index_of_periods_to_change = [
                i for i in range(periods_count)
                if np.random.random() < self.noise_level
            ]

            period = list(period)
            seq = np.array(
                period * periods_count + period[:tail]
            )

            for e in index_of_periods_to_change:
                l, r = e * self.period_size, (e + 1) * self.period_size

                period_copy = period[::]
                period_copy[
                    int(np.random.uniform(high=self.period_size))
                ] = int(np.random.uniform(high=self.alphabet_size))

                seq[l:r] = period_copy

            self.sequences.append(seq)

        if self.path:
            for i, seq in enumerate(self.sequences):
                p = os.path.join(
                    self.path,
                    'seq{}size{}noise_{}alphabet{}period{}.txt'.format(
                        i,
                        self.sequence_size,
                        self.noise_level,
                        self.alphabet_size,
                        self.period_size
                    )
                )

                with open(p, 'a+') as rf:
                    seq = ''.join([chr(97 + e) for e in seq])
                    rf.write(seq)
        else:
            return np.array(self.sequences)


if __name__ == '__main__':
    import os

    for s_size in np.linspace(100, 10000, 10):
        for nl in np.linspace(0.1, 0.9, 7):
            for p_size in np.linspace(2, 21, 5):
                gen = GenerateData(
                    sequence_size=int(s_size),
                    alphabet_size=26,
                    period_size=int(p_size),
                    noise_level=nl,
                    sequences_count=10,
                    path=os.path.join(os.getcwd(), 'data')
                )

                gen.generate()
