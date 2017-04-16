# coding: utf-8

import numpy as np


class PrepareSequence(object):
    def __init__(self, string):
        self.sequence = string
        self.symbolic = string
        self.alphabet = None
        self.alphabet_dict = {}
        self.alphabet_symbol = {}

        if isinstance(self.sequence, str):
            self.sequence = self.normalise()

    def inverse(self):
        return self.sequence[::-1]

    def normalise(self):
        self.sequence = np.array(
            [ord(e) - ord('a') + 1 for e in self.sequence]
        )

        return self.sequence

    def bin_code(self):
        """
        for alphabet [a, b, c] alphabet dict will looks like
         {1: '100', 2: '010', 3: '001'}
        :return: 
        """
        self.alphabet = np.unique(self.sequence)

        for s, n in zip([chr(k + ord('a') - 1) for k in self.alphabet], self.alphabet):
            self.alphabet_symbol[s] = n

        sigm = len(self.alphabet)
        bin_code = []
        for i, e in enumerate(self.alphabet):
            em = [0] * sigm
            em[sigm - 1 - i] = 1
            bin_code.append(em)

        for i in range(len(bin_code)):
            self.alphabet_dict[self.alphabet[i]] = bin_code[i]

        return reduce(lambda r, e: r + self.alphabet_dict[e], self.sequence, [])


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
    s = PrepareSequence('abca')
    print(s.bin_code())
    print(s.alphabet_dict)

    # gen = GenerateData(
    #     sequence_size=100,
    #     alphabet_size=5,
    #     period_size=3,
    #     noise_level=0.2,
    #     sequences_count=20
    # )

    # print(gen.generate())

