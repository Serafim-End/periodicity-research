# coding: utf-8

import math
import bisect

from node import OccurList, Node, OccurNode
from model import Prime
from period import Period

from algorithm import STNRAlgorithm, PeriodCollection


class Edge(object):
    """

    :param T: :type str
    """
    T = None

    # type of st is OccurNode instance

    def __cmp__(self, other):
        if isinstance(other, Edge):
            return self.start_node > other.start_node

    HASH_TABLE_SIZE = 0

    firstEdgesIndex = -1  # type int
    stNodeArray = None  # type list of int
    # edges instance of Edge
    edges = []

    StopSubstringNodeSearch = False  # type: # bool

    def __init__(self, init_first, init_last, parent_node):
        self.first_char_index = init_first
        self.last_char_index = init_last
        self.start_node = parent_node
        self.end_node = Node.Count
        self.value = -1
        self.len = 0
        self.eid = -1
        self.st = None

        if init_first != 0 and init_last != 0 and parent_node != -1:
            Node.Count = Node.Count + 1

    @staticmethod
    def hash(node, c):
        if isinstance(c, str):
            c = ord(c)
        return ((node << 8) + c) % Edge.HASH_TABLE_SIZE

    @classmethod
    def default(self):
        return Edge(init_first=0,
                    init_last=0,
                    parent_node=-1)

    def insert(self):
        i = self.hash(self.start_node, Edge.T[self.first_char_index])

        while Edge.edges[i].start_node != -1:
            i += 1
            i = i % Edge.HASH_TABLE_SIZE

        Edge.edges[i] = Edge.default()
        Edge.edges[i].start_node = self.start_node
        Edge.edges[i].end_node = self.end_node
        Edge.edges[i].first_char_index = self.first_char_index
        Edge.edges[i].last_char_index = self.last_char_index
        Edge.edges[i].value = self.value

    def remove(self):

        i = Edge.hash(self.start_node, Edge.T[self.first_char_index])
        while (Edge.edges[i].start_node != self.start_node or
                Edge.edges[i].first_char_index != self.first_char_index):
            i += 1
            i = i % Edge.HASH_TABLE_SIZE

        while True:
            Edge.edges[i] = Edge.default()
            j = i

            while True:
                i += 1
                i = i % Edge.HASH_TABLE_SIZE

                if Edge.edges[i].start_node == -1:
                    return

                r = Edge.hash(Edge.edges[i].start_node,
                              Edge.T[Edge.edges[i].first_char_index])

                if i >= r > j:
                    continue

                if r > j > i:
                    continue

                if j > i >= r:
                    continue
                break

            Edge.edges[j] = Edge.default()
            Edge.edges[j].start_node = Edge.edges[i].start_node
            Edge.edges[j].end_node = Edge.edges[i].end_node
            Edge.edges[j].first_char_index = Edge.edges[i].first_char_index
            Edge.edges[j].last_char_index = Edge.edges[i].last_char_index
            Edge.edges[j].value = Edge.edges[i].value

    @staticmethod
    def find(node, c):
        """

        :param node: :type: int
        :param c: :type char
        :return: 
        """

        i = Edge.hash(node, c)
        while True:
            if Edge.edges[i].start_node == node:
                if c == Edge.T[Edge.edges[i].first_char_index]:
                    return Edge.edges[i]
            if Edge.edges[i].start_node == -1:
                return Edge.edges[i]
            i += 1
            i = i % Edge.HASH_TABLE_SIZE

    @staticmethod
    def hash_(node):
        return (node << 8) % Edge.HASH_TABLE_SIZE

    @staticmethod
    def FindAll(st_node):
        """

        :param st_node: :type: int
        :return: list of Edge
        """
        result = []
        i = Edge.hash_(st_node)
        while True:
            if Edge.edges[i].start_node == st_node:
                result.append(Edge.edges[i])

            if Edge.edges[i].start_node == -1:
                return result
            i += 1
            i = i % Edge.HASH_TABLE_SIZE

    @staticmethod
    def FindAll1(st_node):
        SuffTree.FindAll1Counter += 1
        idx = Edge.FindIndex(st_node)

        result = []
        if idx < 0:
            return result

        i = idx
        while True:
            if Edge.edges[i].start_node < st_node:
                idx = i + 1
                break
            i -= 1

        i = idx
        while i < len(Edge.edges):
            if Edge.edges[i].start_node > st_node:
                break
            elif Edge.edges[i].start_node == st_node:
                result.append(Edge.edges[i])
            i += 1
        return result

    @staticmethod
    def FindIndex(st_node):
        return bisect.bisect_left(Edge.stNodeArray, st_node)

    @staticmethod
    def FindFirstEdgeIndex():
        i = 0
        while i < len(Edge.edges):
            if Edge.edges[i].start_node != -1:
                Edge.firstEdgesIndex = i
                break
            i += 1

        nEdges = [Edge.default() for i in
                  range(len(Edge.edges) - Edge.firstEdgesIndex + 1)]
        k = Edge.firstEdgesIndex - 1
        c = 0
        while k < len(Edge.edges):
            nEdges[c] = Edge.edges[k]
            k += 1
            c += 1

        Edge.edges = nEdges
        return Edge.firstEdgesIndex

    def SplitEdge(self, s):
        self.remove()
        new_edge = Edge(self.first_char_index,
                        self.first_char_index + s.last_char_index - s.first_char_index,
                        s.origin_node)

        new_edge.insert()
        SuffTree.Nodes[new_edge.end_node].suffix_node = s.origin_node
        self.first_char_index += s.last_char_index - s.first_char_index + 1
        self.start_node = new_edge.end_node
        self.insert()
        return new_edge.end_node

    @staticmethod
    def sort():
        Edge.edges = sorted(Edge.edges)
        Edge.FindFirstEdgeIndex()
        Edge.stNodeArray = [0] * len(Edge.edges)

        i = 0
        while i < len(Edge.edges):
            Edge.stNodeArray[i] = Edge.edges[i].start_node
            i += 1

    def PrefixMatch(self, theSubstring):
        Edge.StopSubstringNodeSearch = False
        edgeStringLength = self.last_char_index - self.first_char_index + 1
        if len(theSubstring) < edgeStringLength:
            Edge.StopSubstringNodeSearch = True
            return theSubstring == Edge.T[
                                   self.first_char_index: len(theSubstring)]

        i = 0
        while i < edgeStringLength:
            if not (Edge.T[self.first_char_index + i] == theSubstring[i]):
                return False
            i += 1

        return True


class EdgeStruct(object):
    def __init__(self, e, tabs, pnValue, pnIndexesIndex=None):
        self.e = e
        self.tabs = tabs
        self.pnValue = pnValue
        self.pnIndexesIndex = pnIndexesIndex
        self.occurStart = [None]
        self.occurLength = [-1]

        self.pnOccurStart = [None]
        self.pnOccurLength = [-1]



class TempSuffix(object):

    def __init__(self, theSuffix, thePosition):
        """
        
        :param theSuffix: :type: str
        :param thePosition: :type; int
        """
        self.ThePosition = thePosition
        self.TheSuffix = theSuffix

    def __cmp__(self, other):
        return self.TheSuffix == other.TheSuffix


class Suffix(object):

    T = None

    def __init__(self, node, start, stop):
        """
        
        :param node: :type: int
        :param start: :type: int 
        :param stop: :type: int
        """
        self.origin_node = node
        self.first_char_index = start
        self.last_char_index = stop

    def Explicit(self):
        return self.first_char_index > self.last_char_index

    def Implicit(self):
        return self.last_char_index >= self.first_char_index

    def Canonize(self):
        if not self.Explicit():
            edge = Edge.find(self.origin_node, self.T[self.first_char_index])
            edge_span = edge.last_char_index - edge.first_char_index

            while edge_span <= (self.last_char_index - self.first_char_index):
                self.first_char_index += edge_span + 1
                self.origin_node = edge.end_node
                if self.first_char_index <= self.last_char_index:
                    edge = Edge.find(edge.end_node,
                                     self.T[self.first_char_index])
                    edge_span = edge.last_char_index - edge.first_char_index


class SuffixHelper(object):
    def FindSubstring(self, theSubstring):
        pass

    def FindAllOccurrences(self, theSubstring):
        pass


class SuffTree(SuffixHelper):
    FindAll1Counter = 0
    PeriodExistCounter = 0
    AddPeriodCounter = 0

    # :param Nodes: :type: Node[]
    Nodes = None

    def __init__(self, T, minThreshold=0.5, tolWin=0, dmax=0,
                 minLengthOfSegment=0, specificPeriod=-1):
        """
        
        :param T: :param: str
        :param minThreshold: :param: double  
        :param tolWin: :param: int
        :param dmax: :param: int
        :param minLengthOfSegment: :param:  double 
        :param specificPeriod: :param: int
        """
        self.ov = ''

        self.T = T
        self.N = len(T)
        self.minThreshold = minThreshold
        self.tolWin = tolWin
        self.dmax = dmax
        self.minLengthOfSegment = minLengthOfSegment
        self.specificPeriod = specificPeriod

        self.occur = OccurList()
        self.periods = []
        self.edht = {}
        self.calculatePeriodCounter = 0
        self.periodColExistCounter = 0
        self.outerLoopCounterBeforeCheck = 0
        self.outerLoopCounterAfterCheck = 0
        self.innerLoopCounter = 0
        self.diffCounter = 0
        self.stn = 0
        self.ovl = []

        self.MakeTree(T, minThreshold)
        # PrintTree(filename);

    def MakeTree(self, T, minThreshold):
        self.minThreshold = minThreshold
        self.T = T
        self.N = len(T)
        Node.Count = 1

        Suffix.T = T
        Edge.T = T
        print T

        SuffTree.Nodes = [Node.default()] * (2 * self.N)
        prime = Prime(int(2 * self.N * 1.1)).next()

        Edge.HASH_TABLE_SIZE = prime
        Edge.edges = [Edge.default()] * prime

        active = Suffix(0, 0, -1)
        for i in range(self.N):
            self.AddPrefix(active, i)

    def AddPrefix(self, active, last_char_index):
        """
        
        :param active: :type: Suffix
        :param last_char_index: :type: int
        :return: 
        """

        # parent_node = None
        last_parent_node = -1

        while True:
            # edge = Edge.default()
            parent_node = active.origin_node

            # first char > last char
            if active.Explicit():
                edge = Edge.find(active.origin_node, self.T[last_char_index])
                if edge.start_node != -1:
                    break
            else:
                edge = Edge.find(active.origin_node, self.T[active.first_char_index])
                span = active.last_char_index - active.first_char_index
                if self.T[edge.first_char_index + span + 1] == self.T[last_char_index]:
                    break

                parent_node = edge.SplitEdge(active)

            new_edge = Edge(init_first=last_char_index,
                            init_last=self.N - 1,
                            parent_node=parent_node)
            new_edge.insert()
            if last_parent_node > 0:
                self.Nodes[last_parent_node].suffix_node = parent_node

            last_parent_node = parent_node

            if active.origin_node == 0:
                active.first_char_index += 1
            else:
                active.origin_node = self.Nodes[active.origin_node].suffix_node
            active.Canonize()

        if last_parent_node > 0:
            self.Nodes[last_parent_node].suffix_node = parent_node
        active.last_char_index += 1
        active.Canonize()

    def FindNode(self, index):
        for n in self.Nodes:
            if n.idx == index:
                return n

        n1 = Node.default()
        n1.idx = index
        return n1

    def PrintTree(self):
        Edge.sort()
        indexes = []
        self.stn = 0
        if self.specificPeriod == -2:
            self.VisitNode4Imp(0, 0, indexes)
            # WritePeriods2Imp(filename)

    def WritePeriods2Imp(self):
        pass

    def VisitNode4Imp(self, tabs, pnvalue, pnIndexes):
        """
        
        :param tabs: :type: int
        :param pnvalue: :type: int
        :param pnIndexes: :type: int[]
        :return: 
        """

        self.MakeEdgeVectorHashtable1()
        s = []  # :type: Stack(EdgeStruct)
        rootOccurSt = [None]  # :type: OccurNode[]
        rootOccurLength = [-1]  # :type: int[]
        edgeCol = Edge.FindAll1(self.stn)  # List(Edge)

        # TODO: check this correctness
        edgeCol = sorted(edgeCol, cmp=self.EdgeComparer)

        for e in edgeCol:
            es = EdgeStruct(e=e, tabs=0, pnValue=0)
            es.pnOccurStart = rootOccurSt
            es.pnOccurLength = rootOccurLength
            s.append(es)

        # edgeCol = None

        while len(s) != 0:
            es = s.pop()

            if es.e.value != -1:
                es.e.st = es.occurStart[0]
                es.e.len = es.occurLength[0]
                if self.IsCalculatePeriod(es.e):
                    self.CalculatePeriodImp(es.e)

                if es.pnOccurStart[0] is None:
                    es.pnOccurStart[0] = es.occurStart[0]
                    es.pnOccurLength[0] = es.occurLength[0]
                else:
                    self.occur.sort(es.pnOccurStart, es.pnOccurLength[0],
                                    es.occurStart, es.occurLength[0])
                    es.pnOccurLength[0] += es.occurLength[0]

            else:

                # myval = 0
                if es.e.last_char_index == (self.N - 1):
                    my_val = self.N - (
                        (es.e.last_char_index - es.e.first_char_index)
                        + 1 + es.pnValue
                    )

                    self.occur.add(
                        value=my_val,
                        pnStart=es.pnOccurStart,
                        pnLength=es.pnOccurLength
                    )

                else:
                    my_val = (
                        (es.e.last_char_index - es.e.first_char_index)
                        + 1 + es.pnValue
                    )

                    es.e.value = my_val
                    s.append(es)
                    self.stn = es.e.end_node
                    eCol = Edge.FindAll1(self.stn)

                    # TODO: check correctness of this
                    eCol = sorted(eCol, cmp=self.EdgeComparer)

                    for e in eCol:
                        es_new = EdgeStruct(
                            e=e,
                            tabs=es.tabs + 1,
                            pnValue=es.e.value
                        )

                        es_new.pnOccurStart = es.occurStart
                        es_new.pnOccurLength = es.occurLength
                        s.append(es_new)

                    # eCol = None

    def MakeEdgeVectorHashtable1(self):
        self.edht[-1] = 0
        s = []  # :type: Stack(EdgeStruct)

        eidCounter = 0
        self.edht[eidCounter] = 0
        eidCounter += 1

        stn = self.stn
        edgeCol = Edge.FindAll1(stn)  # list of Edges

        for e in edgeCol:
            es = EdgeStruct(e=e,tabs=0, pnValue=0)
            es.pnIndexesIndex = 0
            s.append(es)

        while len(s) != 0:
            es = s.pop()

            if es.e.value != -1:
                self.edht[es.pnIndexesIndex] += self.edht[es.e.eid]
                es.e.value = -1
            else:

                if es.e.last_char_index == (self.N - 1):
                    self.edht[es.pnIndexesIndex] += 1
                else:
                    es.e.value = 0
                    self.edht[eidCounter] = 0
                    es.e.eid = eidCounter
                    eidCounter += 1
                    s.append(es)

                    stn = es.e.end_node
                    eCol = Edge.FindAll1(stn)
                    eCol = sorted(eCol, cmp=self.EdgeComparer)

                    for e in eCol:
                        es_new = EdgeStruct(
                            e=e,
                            tabs=0,
                            pnValue=0,
                            pnIndexesIndex=es.e.eid
                        )
                        s.append(es_new)

    def IsCalculatePeriod(self, e):
        """
        :param e: :type: Edge 
        :return: :type: bool
        """
        return True

    def SortEdges(self, eCol):
        """
        FOR FUTURE
        :param eCol: :type: Edge[]
        :return: 
        """
        return None

    def EdgeComparer(self, e1, e2):
        return self.edht[e1.eid] == self.edht[e2.eid]

    def CalculatePeriodImp(self, e):
        """
        
        :param e: :type: Edge
        :return: :type: void
        """

        if STNRAlgorithm.minStrLen > e.value > STNRAlgorithm.maxStrLen:
            return

        current = e.st

        if e.value < 35:
            for i in range(e.len):
                self.ovl.append(current.value)
                current = current.next

            current = e.st
            STNRAlgorithm.CalculatePeriod(self.ovl, e.value)
            self.ovl = None
            return

        last = e.st
        for k in range(e.len):
            last = last.next

        lastOccurValue = last.value
        preDiffValue = -5
        self.calculatePeriodCounter += 1

        for i in range(1, e.len):
            self.outerLoopCounterBeforeCheck += 1
            diffValue = current.next.value - current.value

            if (diffValue < 2 or diffValue < e.value or
                        diffValue > 20 or diffValue == preDiffValue):
                self.diffCounter += 1
                current = current.next
                continue

            p = Period(
                periodValue=diffValue,
                stPos=current.value,
                threshold=0,
                endPos=lastOccurValue
            )

            p.fci = p.stPos
            p.length = e.value
            preDiffValue = diffValue

            if PeriodCollection.exist(p):
                self.periodColExistCounter += 1
                current = current.next
                continue

            self.outerLoopCounterAfterCheck += 1
            p.foundPosCount = 0

            A, C = 0
            B = 0
            sumPerVal = 0
            preSubCurValue = -5
            preStPos = p.stPos
            currStPos = p.stPos

            subCurrent = current

            for j in range(i, e.len):
                self.innerLoopCounter += 1
                A = subCurrent.value - currStPos
                B = round(float(A) / p.periodValue)
                C = A - (p.periodValue * int(B))

                if (-1 * self.tolWin) <= C <= self.tolWin:

                    if round(float((preSubCurValue - currStPos) / p.periodValue)) != B:
                        preSubCurValue = subCurrent.value
                        currStPos = subCurrent.value
                        p.foundPosCount += 1
                        sumPerVal += (p.periodValue + C)
                if (j != e and j < e.len and
                            (subCurrent.next.value - subCurrent.value) >= self.dmax and
                            (currStPos - p.stPos) >= (self.minLengthOfSegment * (len(self.T) - 1))):
                    p.endPos = currStPos
                    break
                subCurrent = subCurrent.next

            y = 0.
            p.avgPeriodValue = (sumPerVal - p.periodValue) / (p.foundPosCount - 1)

            if ((p.endPos + p.length - p.stPos) % int(round(p.avgPeriodValue))) >= e.value:
                y = 1.
            th1 = p.foundPosCount / math.floor((float(p.endPos + p.length - p.stPos) / p.avgPeriodValue) + y)
            p.threshold = th1

            if p.threshold >= self.minThreshold:
                self.AddPeriodCounter += 1
                PeriodCollection.add(p.periodValue, p)
            current = current.next

    def AddOccurNodes(self, eSource, eDest):
        """
        
        :param eSource: :type: Edge 
        :param eDest: :type: Edge
        :return: 
        """

        if eDest.st is None:
            eDest.st = eSource.st
            eDest.len = eSource.len
        else:
            on = eDest.st
            while on.next is not None:
                on = on.next

            on.next = OccurNode(value=eSource.st.value,
                                next=eSource.st.next,
                                previous=None)
            eDest.len += eSource.len

    def StartNode(self, e):
        return e.start_node == self.stn

    def IsPeriodExist(self, pin):
        """
        
        :param pin: :type: Period
        :return:  bool
        """

        self.PeriodExistCounter += 1

        for p in self.periods:
            if p.periodValue == pin.periodValue:
                if p.stPos == pin.stPos:
                    if p.length == pin.length:
                        return True
                    if p.length > pin.length:
                        return True

                elif ((self.T[p.fci: p.length] == self.T[pin.fci, pin.length]) and
                          ((pin.stPos % pin.periodValue) == (p.stPos % pin.periodValue))):
                    if pin.stPos < p.stPos:
                        p.stPos = pin.stPos
                        p.foundPosCount = pin.foundPosCount
                        p.threshold = pin.threshold
                        p.fci = pin.fci
                        p.length = pin.length

                    return True
        return False

    def FindEdge(self, endNode):
        """
        
        :param endNode: :type: int
        :return: 
        """

        for e in Edge.edges:
            if e.end_node == endNode:
                return e
        return None

    def GetEdgesWithSource(self, sourceNode):
        """
        
        :param sourceNode: :type: int
        :return: Edge[]
        """
        originEdges = []
        for edge in Edge.edges:
            if edge.start_node == sourceNode:
                originEdges.append(edge)
        return originEdges

    def FindSubstring(self, theSubstring):
        """
        
        :param theSubstring: :type: str
        :return: 
        """
        if theSubstring is None:
            return False

        substringEndNode = self.FindSubstringNode(theSubstring, 0, self.GetEdgesWithSource(0))
        if substringEndNode != -1:
            return True
        else:
            return False

    def FindSubstringNode(self, theSubstring, currentNode, currentChildEdges):
        """
        
        :param theSubstring: :type: str
        :param currentNode: :type: int
        :param currentChildEdges: :type: Edge[]
        :return: :type: int
        """

        if theSubstring == '':
            return currentNode
        for edge in currentChildEdges:
            if edge.PrefixMatch(theSubstring):
                if Edge.StopSubstringNodeSearch:
                    Edge.StopSubstringNodeSearch = False
                    return edge.end_node

                return self.FindSubstringNode(
                    theSubstring.Substring(edge.last_char_index - edge.first_char_index + 1),
                    edge.end_node,
                    self.GetEdgesWithSource(edge.end_node)
                )

        return -1

    def FindAllOccurrences(self, theSubstring):
        """
        
        :param theSubstring: :type: str
        :return: :type: int[]
        """
        if theSubstring is None:
            return []

        substringEndNode = self.FindSubstringNode(theSubstring, 0, self.GetEdgesWithSource(0))
        childEdges = self.GetEdgesWithSource(substringEndNode)
        substringIndexList = []

        if substringEndNode == -1:
            return []

        if len(childEdges) == 0:
            substringIndexList.append(self.T.index(theSubstring))
            return substringIndexList

        substringIndexList = self.FindLeafIndexes(
            substringEndNode,
            substringIndexList,
            childEdges
        )

        return substringIndexList

    def FindLeafIndexes(self, originNode, currentIndexList, currentEdges):
        """
        
        :param originNode: :type: int
        :param currentIndexList: :type: int[]
        :param currentEdges: type: Edge[]
        :return: 
        """

        localList = []

        for edge in currentEdges:
            loopEdgeList = self.GetEdgesWithSource(edge.end_node)
            if len(loopEdgeList) == 0:
                localList.append(edge.value)
            else:
                localList += self.FindLeafIndexes(edge.end_node, localList, loopEdgeList)
        return localList
