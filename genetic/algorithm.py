# coding: utf-8

import numpy as np

from genetic.model import Organism, Population

import settings


def generate_artificial_sequence(n, sequence_length, alphabet_size):

    period = np.random.choice(alphabet_size, n, replace=False).tolist()
    k = np.ceil(sequence_length / n)
    tail = int(sequence_length - k * n)
    return np.array(period), np.array(period * k + period[:tail])


class KorotkovGenetic(object):
    def __init__(self, S, alphabet_size, n):
        self.S = S

        self.period, self.S1 = generate_artificial_sequence(
            n, S.size, alphabet_size
        )

        self.alphabet_size = alphabet_size
        self.n = n

        self.population = Population()

    def initialize(self, population_size):
        while self.population.size < population_size:
            o = Organism(self.n, self.S, self.S1, self.alphabet_size)
            if self.population and o.cmp_all(self.population):
                self.population.organisms.append(o)

    def evolute(self, number_iterations, p_increase_delay):
        """
        :param number_iterations: stopping criteria - limit of operations
        :param p_increase_delay: possible increase delay
         - number of steps without Fmax increasing
        :return: 
        """

        curr_iter = 0
        curr_pid = 0
        while curr_iter < number_iterations and curr_pid < p_increase_delay:
            curr_iter += 1

            cut_portion = int(
                settings.POPULATION_PERCENT * self.population.size
            )

            parents = self.population.sort()[:cut_portion].copy()

            Fmax_sum = reduce(lambda r, e: r + e.Fmax, parents, 0)
            probabilities = [float(o.Fmax) / Fmax_sum for o in parents]

            new_population = []
            for i in range(len(parents)):
                mam, dad = np.random.choice(
                    parents,
                    size=2,
                    replace=False,
                    p=probabilities
                )

                ch1, ch2 = self.crossover(mam, dad)
                ch1, ch2 = self.crossing_diff(ch1, ch2)
                ch = self.killing(ch1, ch2)
                new_population.append(ch)

            print(len(new_population))

    def crossover(self, a, b):

        def flatten(o, back=False):
            if not back:
                o.m = np.reshape(o.m, (o.m.shape[0] * o.m.shape[1],))
            else:
                o.m = np.reshape(
                    o.m,
                    (settings.SIZE_PERIOD_TYPES,
                     o.m.shape[0] / settings.SIZE_PERIOD_TYPES)
                )

        la, ra = np.random.randint(0, a.m.shape[0] * a.m.shape[1], 2)
        lb, rb = np.random.randint(0, b.m.shape[0] * b.m.shape[1], 2)

        flatten(a), flatten(b)
        b_m_copy = b.m.copy()

        if la > ra:
            b_m_copy[la:], b_m_copy[:ra] = a.m[la:], a.m[:ra]
        else:
            b_m_copy[la:ra] = a.m[la:ra]

        if lb > rb:
            a.m[lb:], a.m[:rb] = b.m[lb:], b.m[:rb]
        else:
            a.m[lb: rb] = b.m[lb: rb]

        b.m = b_m_copy

        def mutate_rnd(o):
            # make mutations here
            mutation = np.random.choice(
                a=range(o.m.shape[0]),
                size=0.01 * o.m.shape,
                p=0.01 * np.ones(o.m.shape[1])
            )

            for e in mutation:
                o.m[e] = np.random.uniform(-1, 1)
            return o

        def mutate_each(o):
            """
            mutate whole matrix not so quick
            :param o: organism
            :return: organism
            """
            sign = np.sign(np.random.random() * 2 - 1)
            p2 = np.random.uniform(0.001, 0.03)
            o.m += sign * p2 * o.m
            return o

        a, b = mutate_rnd(a), mutate_rnd(b)
        flatten(a, back=True), flatten(b, back=True)
        return a, b

    def crossing_diff(self, a, b):
        alpha = np.random.uniform()

        _x = alpha * a.m
        _y = alpha * b.m

        w = _x + b.m - _y
        v = a.m - _x + _y

        a.m, b.m = w, v
        return a, b

    def killing(self, a, b):

        if a.Fmax > b.Fmax:
            return a
        else:
            return b


def korotkov_algorithm(S, alphabet_size):

    N = S.size

    high_boundary = alphabet_size if alphabet_size > 200 else 200

    for n in range(2, high_boundary):
        S1 = generate_artificial_sequence(n, N, alphabet_size)


if __name__ == '__main__':

    alphabet_size = 7
    n = 5
    artificial_n = 4
    N = 150
    pS, S = generate_artificial_sequence(n, N, alphabet_size)

    genetic = KorotkovGenetic(S, alphabet_size, n)

    o1 = Organism(n, S, genetic.S1, alphabet_size)
    o2 = Organism(n, S, genetic.S1, alphabet_size)

    print(genetic.crossover(o1, o2))

    # pS1, S1 = generate_artificial_sequence(artificial_n, N, alphabet_size)
    #
    # print('S: {}'.format(S))
    # print('S1: {}'.format(S1))
    # initialize(3, S, S1, 5)
    # print generate_weight_matrix(n, S, S1, alphabet_size)
    # print generate_artificial_sequence(4, 15, 5)
