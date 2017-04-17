# coding: utf-8

import math

from period import CPeriod


class PeriodCollection(object):

    # {
    #       period_value1: {
    #           start_position1: CPeriod1,
    #           start_position2: CPeriod2,
    #       },
    #       period_value_2: ...,
    #       ...
    # }

    periodList = {}

    @staticmethod
    def add(periodVal, period):
        """
        :param periodVal: :type: int
        :param period: :type Period
        :return: 
        """

        def contain_key(d, k):
            return k in d.keys()

        if contain_key(PeriodCollection.periodList, periodVal):

            inner_list = PeriodCollection.periodList[period.periodValue]
            keys_to_remove = []

            for sp in inner_list.keys():
                if ((sp % period.periodValue == period.stPos % period.periodValue) and
                        (STNRAlgorithm.s[period.stPos:period.length].startswith(STNRAlgorithm.s[sp: PeriodCollection.periodList[periodVal][sp]].length))):
                    keys_to_remove.append(sp)

            for i in range(len(keys_to_remove)):
                del inner_list[keys_to_remove[i]]

            inner_list[period.stPos] = period

        else:
            keys_to_remove = []
            for p in PeriodCollection.periodList.keys():

                if p <= periodVal:
                    continue

                if p % period.periodValue == 0:
                    if (contain_key(PeriodCollection.periodList[p], period.stPos) and
                                PeriodCollection.periodList[p][period.stPos].length <= period.length):
                        del PeriodCollection.periodList[p][period.stPos]
                        if len(PeriodCollection.periodList[p]) == 0:
                            keys_to_remove.append(p)

            for e in keys_to_remove:
                del PeriodCollection.periodList[e]

            inner_list = {period.stPos: period}
            PeriodCollection.periodList[periodVal] = inner_list

    @staticmethod
    def exist(period):
        for p in PeriodCollection.periodList.keys():

            if p > period.periodValue:
                break

            if period.periodValue % p == 0:
                for j in PeriodCollection.periodList[p].keys():
                    if j > period.stPos:
                        break

                    if (PeriodCollection.periodList[p][j].length >= period.length and
                            PeriodCollection.periodList[p][j].endPos >= period.stPos and
                            ((j % p == period.stPos % p) or
                                ((j + PeriodCollection.periodList[p][j].length - 1) >= period.stPos))):
                        return True
            return False

    def PeriodValueExist(self, p):
        for i in PeriodCollection.periodList.keys():

            if i > p.periodValue:
                break

            if (p.periodValue % i) == 0:
                for j in PeriodCollection.periodList[i].keys():
                    if j > p.stPos:
                        break

                    if j % i == p.stPos % i:
                        return True
        return False



class STNRAlgorithm(object):
    s = ''

    minTh = 0.7
    minLengthSegment = 0.9
    minPeriod = 1
    maxPeriod = 500
    minStrLen = 1
    maxStrLen = 100000
    tolWin = 0
    dmax = 10000
    periodCollection = []
    preCountPerCol = len(periodCollection)
    candPerCount = 0
    addPerCount = 0
    occVecCount = 0
    occVecAddCount = 0

    def __init__(self):
        self.periodCollection = []
        self.preCountPerCol = len(self.periodCollection)
        self.candPerCount = 0
        self.addPerCount = 0
        self.occVecCount = 0
        # self.s = 'abcdeabcdeabcde' + '$'

    def CalculatePeriod(self, occur, strlen):
        """
        
        :param occur: :type: int[]
        :param strlen: :type: int
        :return: 
        """
        self.occVecCount += 1
        if strlen < self.minStrLen or strlen > self.maxStrLen:
            return

        self.candPerCount += (len(occur) - 1)
        preCountPerCol = len(self.periodCollection)
        prePer = -5

        for i in range(len(occur)):
            p = CPeriod(st=occur[i], currSt=0,p=0, sumP=0, avgP=0,
                        lastOccur=occur[i], strlen=strlen, count=0)

            if i < (len(occur) - 1):
                p.p = occur[i + 1] - occur[i]
                if p.p > self.dmax:
                    y1 = 0

                    for k in range(self.preCountPerCol, len(self.periodCollection)):
                        A1 = occur[i] - self.periodCollection[k].currSt
                        B1 = round(float(A1) / self.periodCollection[k].p)
                        C1 = A1 - (self.periodCollection[k].p * int(B1))
                        if (-1 * self.tolWin) <= C1 <= self.tolWin:
                            if (round(float((self.periodCollection[k].preValidVal - self.periodCollection[k].currSt) /
                                                        self.periodCollection[k].p)) != B1):
                                self.periodCollection[k].preValidVal = occur[i]
                                self.periodCollection[k].currSt = occur[i]
                                self.periodCollection[k].sumP += self.periodCollection[k].p + C1
                                self.periodCollection[k].lastOccur = occur[i]
                                self.periodCollection[k].count += 1
                        avgPeriodValue = float(self.periodCollection[k].sumP - self.periodCollection[k].p) / (self.periodCollection[k].count - 1)
                        self.periodCollection[k].avgP = round(avgPeriodValue, 1)

                        if ((self.periodCollection[k].lastOccur +
                                self.periodCollection[k].strlen -
                                self.periodCollection[k].st) % int(round(avgPeriodValue))) >= self.periodCollection[k].strlen:
                            y1 = 1
                        else:
                            y1 = 0

                        self.periodCollection[k].th = (self.periodCollection[k].count /
                                                       math.floor((float(self.periodCollection[k].lastOccur +
                                                                         self.periodCollection[k].strlen -
                                                                         self.periodCollection[k].st) / avgPeriodValue) + y1))
                        if (self.periodCollection[k].th < self.minTh or (
                                    (self.periodCollection[k].lastOccur +
                                         self.periodCollection[k].strlen -
                                         self.periodCollection[k].st) < (self.minLengthSegment * len(self.s)))):
                            self.periodCollection.remove(self.periodCollection[k])
                            k -= 1

                    preCountPerCol = len(self.periodCollection)
                    prePer = -5
                    continue

            if ((p.p != 0) and (prePer != p.p) and
                        (occur[occur.Length - 1] + strlen - p.st) > (self.minLengthSegment * len(self.s)) and not self.AlreadyThere(p)):
                p.currSt = p.st
                self.periodCollection.append(p)
                self.addPerCount += 1

            prePer = p.p

            for j in range(self.preCountPerCol, len(self.periodCollection)):
                A = occur[i] - self.periodCollection[j].currSt
                B = round(float(A) / self.periodCollection[j].p)
                C = A - (self.periodCollection[j].p * int(B))

                if (-1 * self.tolWin) <= C <= self.tolWin:
                    if (round(float(self.periodCollection[j].preValidVal - self.periodCollection[j].currSt) / self.periodCollection[j].p)) != B:
                        self.periodCollection[j].preValidVal = occur[i]
                        self.periodCollection[j].currSt = occur[i]
                        self.periodCollection[j].sumP += (self.periodCollection[j].p + C)
                        self.periodCollection[j].lastOccur = occur[i]
                        self.periodCollection[j].count += 1

            y = 0
            for i in range(preCountPerCol, len(self.periodCollection)):
                avgPeriodValue = float(self.periodCollection[i].sumP - self.periodCollection[i].p) / (self.periodCollection[i].count - 1)
                self.periodCollection[i].avgP = round(avgPeriodValue, 1)

                if ((self.periodCollection[i].lastOccur +
                         self.periodCollection[i].strlen -
                         self.periodCollection[i].st) % (int(round(avgPeriodValue))) >= self.periodCollection[i].strlen):
                    y = 1
                else:
                    y = 0

                self.periodCollection[i].th = (self.periodCollection[i].count /
                                               math.floor((float(self.periodCollection[i].lastOccur +
                                                                 self.periodCollection[i].strlen -
                                                                 self.periodCollection[i].st) / avgPeriodValue) + y))
                if (self.periodCollection[i].th < self.minTh or
                        ((self.periodCollection[i].lastOccur +
                              self.periodCollection[i].strlen -
                              self.periodCollection[i].st) < (self.minLengthSegment * len(self.s)))):
                    self.periodCollection.remove(self.periodCollection[i])
                    i -= 1

    def AlreadyThere(self, p):
        """
        
        :param p: :type: CPeriod
        :return: :type: bool
        """

        if (p.strlen > p.p or
                    p.p < self.minPeriod or
                    p.p > self.maxPeriod):
            return True
        else:
            # return False

            for i in range(self.preCountPerCol):
                if(self.periodCollection[i].p == p.p or
                                p.p % self.periodCollection[i].p == 0):
                    if p.st == self.periodCollection[i].st:
                        return True

                    if (p.st % p.p == self.periodCollection[i].st % self.periodCollection[i].p and
                                p.strlen <= self.periodCollection[i].strlen):

                        k = 0
                        for k in range(p.strlen):
                            if self.s[p.st + k] != self.s[self.periodCollection[i].st + k]:
                                break
                        if k == p.strlen:
                            return True
                        else:
                            continue
        return False
