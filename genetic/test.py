# coding: utf-8


class CMPTEST(object):

    def __init__(self, a):
        self.a = a

    def __cmp__(self, other):
        return self.a > other.a

    def cmp_all(self, others):
        return all([self > o for o in others])


l = [CMPTEST(1), CMPTEST(2), CMPTEST(4),
     CMPTEST(10), CMPTEST(100), CMPTEST(13)]

CMPTEST(15).cmp_all(l)


with open() as rf:
    rf.write()