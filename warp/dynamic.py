# coding: utf-8

import numpy as np


def dynamic_time_warp(seq_A, seq_B, d=lambda x, y: abs(x - y)):
    # create the cost matrix
    num_rows, num_cols = seq_A.size, seq_B.size

    cost = np.zeros((num_rows, num_cols))
    # initialize the first row and column
    cost[0, 0] = d(seq_A[0], seq_B[0])
    for i in range(1, num_rows):
        cost[i, 0] = cost[i - 1, 0] + d(seq_A[i], seq_B[0])

    for j in range(1, num_cols):
        cost[0, j] = cost[0, j - 1] + d(seq_A[0], seq_B[j])
    # fill in the rest of the matrix
    for i in range(1, num_rows):
        for j in range(1, num_cols):
            cost[i, j] = d(seq_A[i], seq_B[j]) + np.min([
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1]
            ])
    return cost
