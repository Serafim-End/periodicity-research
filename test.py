# coding: utf-8

import unittest

import numpy as np

from generate_data import GenerateData
from warp.algorithm import WARPAlgorithm


class TestSystem(object):

    def __init__(self, noises, sequence_sizes, alphabet_sizes,
                 period_sizes, sequences_counts, verbose=True):

        self.verbose = verbose
        self.noises = np.linspace(0.1, 0.9, 9) if not noises else noises

        if sequence_sizes:
            self.sequence_sizes = sequence_sizes
        else:
            self.sequence_sizes = np.linspace(100, 10000, 10)

        self.alphabet_sizes = (np.linspace(10, 2000, 5) if not alphabet_sizes
                               else alphabet_sizes)

        self.period_sizes = (np.linspace(5, 1000, 5) if not period_sizes
                             else period_sizes)

        if self.period_sizes.size != self.alphabet_sizes.size:
            raise Exception('sizes of period_sizes and alphabet_sizes'
                            ' should be equal')

        self.sequences_count = (100 if not sequences_counts
                                else sequences_counts)

    def make_test(self, algo=WARPAlgorithm):

        correct = 0.
        experiments = (self.noises.size *
                       self.sequence_sizes.size *
                       self.alphabet_sizes.size *
                       self.period_sizes.size *
                       self.sequences_count)

        for noise_level in self.noises:
            for sequence_size in self.sequence_sizes:
                for period_size, alphabet_size in zip(self.period_sizes,
                                                      self.alphabet_sizes):

                    if period_size > alphabet_size:
                        print('It is not OK. period size should be less'
                              ' or equal to alphabet size')

                    g = GenerateData(
                        sequence_size=sequence_size,
                        alphabet_size=alphabet_size,
                        period_size=period_size,
                        noise_level=noise_level,
                        sequences_count=self.sequences_count
                    )
                    for s in g.generate():
                        predicted = algo(s).get_best_candidates()
                        if self.verbose:
                            print(predicted, period_size)
                        for pred in predicted:
                            if (pred % period_size) == 0:
                                correct += 1

        return correct, correct / experiments


class AlgorithmsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_warp_algorithm(self):

        t = TestSystem(
            noises=np.array([0.2]),
            sequence_sizes=np.array([100]),
            alphabet_sizes=np.array([10]),
            period_sizes=np.array([5]),
            sequences_counts=10,
        )

        correct, percent = t.make_test(algo=WARPAlgorithm)

        self.assertTrue(correct / percent > 0, 'no experiments')
        self.assertTrue(correct > 0, 'no correct answers')
        self.assertTrue(
            percent > 0.5,
            'faild experiment, lower than 50% correct,'
            ' just: {}'.format(percent)
        )

    def test_stnr_algorithm(self):
        """
        STNR Algorithm tests should be here
        
        Suffix Tree Noise Resilient algorithm should contain
         get_best_candidates method
        
        :return: statistics after testing
        """
        from stnr.algorithm import STNRAlgorithm

        t = TestSystem(
            noises=np.array([0.1, 0.2, 0.3, 0.4]),
            sequence_sizes=np.array([100, 500, 1000, 3000]),
            alphabet_sizes=np.array([10, 50, 100, 300]),
            period_sizes=np.array([5, 25, 50, 100]),
            sequences_counts=10,
        )

        # algo - implementation of Suffix Tree Noise Resilient Algorithm
        correct, percent = t.make_test(algo=STNRAlgorithm)

        self.assertTrue(correct / percent > 0, 'no experiments')
        self.assertTrue(correct > 0, 'no correct answers')
        self.assertTrue(
            percent > 0.5,
            'faild experiment, lower than 50% correct,'
            ' just: {}'.format(percent)
        )
