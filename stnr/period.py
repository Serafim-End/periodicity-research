# coding: utf-8


class CPeriod(object):
    """
    CPeriod because it chould be comparable
    """

    def __init__(self, st, currSt, p, sumP, avgP,
                 lastOccur, strlen=0, th=0,
                 preValidVal=-200, count=0):

        self.p = p
        self.st = st
        self.currSt = currSt
        self.sumP = sumP
        self.avgP = avgP
        self.lastOccur = lastOccur
        self.strlen = strlen
        self.th = th
        self.preValidVal = preValidVal
        self.count = count

    def __cmp__(self, other):
        return self.p > other.p


class Period(object):
    def __init__(self, periodValue, stPos, endPos, foundPosCount=0,
                 avgPeriodValue=0, threshold=0):
        """
        :param periodValue: :type: int
        :param stPos: :type; int
        :param threshold: :type: double
        :param foundPosCount: :type: int
        :param avgPeriodValue: :type: double
        :param endPos: :type: int
        """
        self.fci = 0
        self.length = 0
        self.periodValue = periodValue
        self.stPos = stPos
        self.threshold = threshold
        self.foundPosCount = foundPosCount
        self.avgPeriodValue = avgPeriodValue
        self.endPos = endPos

    def __cmp__(self, other):
        return self.periodValue.CompareTo(other.periodValue)


class CPeriodCollection(object):

    # {
    #       period_value1: {
    #           start_position1: CPeriod1,
    #           start_position2: CPeriod2,
    #       },
    #       period_value_2: ...,
    #       ...
    # }

    periodList = {}

    def add(self, period):
        """
        :param period: :type CPeriod
        :return: 
        """

        def contain_key(d, k):
            return k in d.keys()

        if contain_key(CPeriodCollection.periodList, period.p):

            inner_list = CPeriodCollection.periodList[period.p]
            keys_to_remove = []

            for sp in inner_list.keys():
                if ((sp % period.p == period.st % period.p) and
                        (period.lastOccur >= CPeriodCollection.periodList[period.p][sp].lastOccur)):
                    keys_to_remove.append(sp)

            for i in range(len(keys_to_remove)):
                del inner_list[keys_to_remove[i]]

            inner_list[period.st] = period

        else:
            keys_to_remove = []
            for p in CPeriodCollection.periodList.keys():

                if p <= period.p:
                    continue

                if p % period.p == 0:
                    if (contain_key(CPeriodCollection.periodList[p], period.st) and
                                CPeriodCollection.periodList[p][period.st].strlen <= period.strlen):
                        del CPeriodCollection.periodList[p][period.st]
                        if len(CPeriodCollection.periodList[p]) == 0:
                            keys_to_remove.append(p)

            for e in keys_to_remove:
                del CPeriodCollection.periodList[e]

            inner_list = {period.st: period}
            CPeriodCollection.periodList[period.p] = inner_list

    def exist(self, period):
        for p in CPeriodCollection.periodList.keys():

            if p > period.p:
                break

            if period.p % p == 0:
                for j in CPeriodCollection.periodList[p].keys():
                    if j > period.st:
                        break

                    if (CPeriodCollection.periodList[p][j].strlen >= period.strlen and
                            CPeriodCollection.periodList[p][j].lastOccur >= period.st and
                            ((j % p == period.st % p) or
                                ((j + CPeriodCollection.periodList[p][j].strlen - 1) >= period.st))):
                        return True
            return False
