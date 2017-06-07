# coding: utf-8


class Node(object):
    Count = 0

    def __init__(self, idx, suffix_node):
        """
        :param idx: :type: int
        :param suffix_node: :type: int
        """
        self.idx = idx
        self.suffix_node = suffix_node

    @classmethod
    def default(self):
        return Node(idx=0, suffix_node=-1)


class OccurNode(object):
    def __init__(self, value=0, next=None, previous=None):
        """
        :param value: :type: int
        :param next: :type: OccurNode
        :param previous: :type: OccurNode
        """
        self.value = value
        self.next = Node.default() if next is None else next
        self.previous = Node.default() if previous is None else previous


class OccurList(object):
    def __init__(self):
        """
        Length :type: int
        firstNode :type: OccurNode
        currentNode :type: OccurNode
        """
        self.Length = 0
        self.firstNode = OccurNode()
        self.currentNode = OccurNode()

    def add(self, value, pnStart, pnLength):
        """

        :param value:  :type int
        :param pnStart: :type OccurNode[]
        :param pnLength: :type: int[]
        :return: 
        """

        n = OccurNode()
        n.value = value

        if pnStart[0] is None:
            self.currentNode.next = n
            n.previous = self.currentNode
            self.currentNode = n
            pnStart[0] = n
            pnLength[0] = 1

        else:
            flag = False
            k = pnStart[0]

            i = 1
            while i <= pnLength[0]:
                if k.value > value:
                    n.previous = k.previous
                    n.next = k
                    k.previous.next = n
                    k.previous = n
                    if k == pnStart[0]:
                        pnStart[0] = n

                    flag = True
                    break

                if i != pnLength[0]:
                    k = k.next

                i += 1

            if not flag:
                k.next = n
                n.previous = k
                self.currentNode = n
            pnLength[0] += 1

        self.Length += 1

    def sort(self, pnOccurStart, pnOccurLength, occurStart, occurLength):
        """
        :param pnOccurStart: :type: OccurNode[]
        :param pnOccurLength: :type: int
        :param occurStart: :type: OccurNode[]
        :param occurLength: :type: int
        :return: 
        """

        flag = False
        prePnOccSt = pnOccurStart[0]
        currOccSt = occurStart[0]
        currPnOccSt = pnOccurStart[0]
        j = 1

        i = 1
        while i <= occurLength:
            temp = currOccSt.next

            while j <= pnOccurLength:
                if currPnOccSt.value > currOccSt.value:
                    if currOccSt.next is not None:
                        currOccSt.next.previous = currOccSt.previous
                    else:
                        flag = True
                    currOccSt.previous.next = currOccSt.next
                    currOccSt.previous = currPnOccSt.previous
                    currOccSt.previous.next = currOccSt
                    currOccSt.next = currPnOccSt
                    currPnOccSt.previous = currOccSt
                    if currPnOccSt == prePnOccSt:
                        pnOccurStart[0] = currOccSt
                        prePnOccSt = currOccSt

                    break

                currPnOccSt = currPnOccSt.next
                j += 1

            currOccSt = temp
            i += 1

        if flag:
            while currPnOccSt.next is not None:
                currPnOccSt = currPnOccSt.next
            self.currentNode = currPnOccSt
