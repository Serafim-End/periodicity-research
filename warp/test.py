# coding: utf-8

from sequence import PrepareSequence
from algorithm import WARPAlgorithm, WARPPoorAlgorithm

if __name__ == '__main__':
    from pprint import pprint
    o = PrepareSequence('abcabcabca')
    s = o.sequence

    pprint(WARPPoorAlgorithm(s).get_certain_period())
    pprint(WARPAlgorithm(s).get_certain_period())
