# coding: utf-8

from sequence import PrepareSequence
from algorithm import CONVAlgorithm


if __name__ == '__main__':
    from pprint import pprint

    s = PrepareSequence('abcabbabcb')
    # s = PrepareSequence('acccabb')
    # s = PrepareSequence('cabccbacd')

    obj = CONVAlgorithm(s.sequence, verbose=True)
    pprint(obj.get_candidates())
    print(obj.get_best_candidates())

