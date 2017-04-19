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


if __name__ == '__main__':
    s = PrepareSequence('abca')
    print(s.bin_code())
    print(s.alphabet_dict)
