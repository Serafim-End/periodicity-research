# coding: utf-8

import numpy as np


class GenerateData(object):

    def __init__(self, sequence_size, alphabet_size, period_size,
                 noise_level, sequences_count):

        self.sequence_size = sequence_size
        self.alphabet_size = alphabet_size
        self.period_size = period_size
        self.noise_level = noise_level
        self.sequences_count = sequences_count

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

        return np.array(self.sequences)


if __name__ == '__main__':
    gen = GenerateData(
        sequence_size=100,
        alphabet_size=5,
        period_size=3,
        noise_level=0.2,
        sequences_count=20
    )

    print(gen.generate())
