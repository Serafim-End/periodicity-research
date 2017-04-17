# coding: utf-8

import math


class Prime(object):

    def __init__(self, lowerBound=0):
        self.current = lowerBound

    def next(self):
        while not Prime.isPrime(self.current):
            self.current += 1
        self.current += 1
        return self.current - 1

    @staticmethod
    def isPrime(n):
        if n < 2:
            return False

        i = 2
        while i <= int(math.sqrt(float(n))):
            if n == i * (n / i):
                return False
            i += 1

        return True
