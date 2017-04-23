# coding: utf-8

import numpy as np
from fastdtw import fastdtw
from dynamic import dynamic_time_warp
from scipy.spatial.distance import hamming


class WARPAlgorithm(object):
    """
    WARP algorithm implementation based on fastDTW
    """
    def __init__(self, sequence):
        self.sequence = sequence
        self.candidates = []
        self.min_dist = np.inf
        self.size = self.sequence.size

    def get_candidates(self):
        """
        implementation of WARP algorithm with
         fast Dynamic Time Warping techology
        :return: list of candidates where each candidate is
         (period_size, distance, path in cost matrix)
        """
        for i in range(1, len(self.sequence) / 2):
            distance, path = fastdtw(
                np.array(self.sequence[i:]),
                np.array(self.sequence[:self.size - i]),
                dist=hamming
            )

            self.candidates.append((i, distance, path))
        return self.candidates

    def get_sorted_candidates(self):
        """
        sorted by distance candidates
        :return: 
        """
        return sorted(self.get_candidates(), key=lambda key: key[1])

    def get_certain_period(self):
        """
        :return: period length = the best candidate
        """
        candidates = self.get_sorted_candidates()
        return int(candidates[0][0])

    def get_best_candidates(self):
        """
        boundary: in every second period is possible to make a mistake
        :return: list of candidates
        """
        return [c[0] for c in self.get_sorted_candidates()
                if c[1] < 0.5 * (self.size / c[0])]


class WARPPoorAlgorithm(WARPAlgorithm):
    def __init__(self, sequence):
        super(WARPPoorAlgorithm, self).__init__(sequence)

    def get_candidates(self):
        for i in range(1, len(self.sequence) / 2):
            cost = dynamic_time_warp(
                np.array(self.sequence[i:]),
                np.array(self.sequence[:self.size - i]),
                d=hamming
            )

            distance = cost[-1, -1]
            self.candidates.append((i, distance))
        return np.array(self.candidates)
