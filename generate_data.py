# coding: utf-8

import numpy as np


class GenerateData(object):

    def __init__(self, sequence_size, alphabet_size, period_size,
                 noise_level, sequences_count, path=None):

        self.sequence_size = sequence_size
        self.alphabet_size = alphabet_size
        self.period_size = period_size

        self.segment_noise_level = noise_level
        self.period_noise_level = noise_level


        self.sequences_count = sequences_count
        self.path = path

        self.sequences = []

    def generate(self, noise_type='replace'):

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
                if np.random.random() < self.segment_noise_level
            ]

            period = list(period)

            noise_seq = []

            for e in range(periods_count):

                if e not in index_of_periods_to_change:
                    noise_seq.append(period)
                    continue

                random_change_in_period = [
                    el for el in range(self.period_size)
                    if np.random.random() < self.period_noise_level
                ]

                period_copy = period[::]

                if noise_type == 'replace':

                    for el in random_change_in_period:
                        period_copy[el] = int(np.random.uniform(high=self.alphabet_size))

                    noise_seq.append(period_copy)

                elif noise_type == 'remove':

                    count = 0
                    for el in random_change_in_period:
                        period_copy.pop(el - count)
                        count += 1

                    noise_seq.append(period_copy)

                elif noise_type == 'insert':

                    count = 0
                    for el in random_change_in_period:
                        period_copy.insert(
                            el + count,
                            int(np.random.uniform(high=self.alphabet_size))
                        )
                        count += 1

                    noise_seq.append(period_copy)

                elif noise_type == 'mix':

                    c_add, c_del = 0, 0
                    for el in random_change_in_period:
                        local_noise_type = np.random.choice([0, 1, 2])
                        if local_noise_type == 0:
                            period_copy[el - c_del + c_add] = int(np.random.uniform(high=self.alphabet_size))

                        if local_noise_type == 1:
                            period_copy.pop(el - c_del + c_add)
                            c_del += 1

                        if local_noise_type == 2:
                            period_copy.insert(
                                el - c_del + c_add,
                                int(np.random.uniform(high=self.alphabet_size))
                            )
                            c_add += 1

            gen_seq = []
            for p in noise_seq:
                gen_seq += p
            gen_seq += period[:tail]

            self.sequences.append(np.array(gen_seq))

        if self.path:
            for i, seq in enumerate(self.sequences):
                p = os.path.join(
                    self.path,
                    'seq{}size{}noise_{}alphabet{}period{}noise_type{}.txt'.format(
                        i,
                        self.sequence_size,
                        max(self.segment_noise_level, self.period_noise_level),
                        self.alphabet_size,
                        self.period_size,
                        noise_type
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
        for nl in np.linspace(0.1, 0.9, 10):
            for p_size in np.linspace(2, 21, 10):
                gen = GenerateData(
                    sequence_size=int(s_size),
                    alphabet_size=26,
                    period_size=int(p_size),
                    noise_level=nl,
                    sequences_count=10,
                    path=os.path.join(os.getcwd(), 'data')
                )

                gen.generate(noise_type='insert')
