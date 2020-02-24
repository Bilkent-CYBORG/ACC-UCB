import numpy as np
from binarytree import Node

from Hypercube import Hypercube
from typing import List

"""
This represents a node as discussed in the paper.
"""


class UcbNode:
    hypercube_list: List[Hypercube]

    # The node obj is simply used to visualize the tree

    def __init__(self, parent_node, h, hypercube_list):
        self.parent_node = parent_node
        self.h = h
        self.hypercube_list = hypercube_list
        self.dimension = self.hypercube_list[0].get_dimension()

    def reproduce(self):
        """
        This fun creates N new nodes and assigns regions (i.e. hypercubes) to them.
        :return: A list of the N new nodes.
        """
        if len(self.hypercube_list) == 1:
            new_hypercubes = []
            new_hypercube_length = self.hypercube_list[0].length / 2
            old_center = self.hypercube_list[0].center
            num_new_hypercubes = 2 ** self.dimension
            for i in range(num_new_hypercubes):
                center_translation = np.fromiter(
                    map(lambda x: new_hypercube_length / 2 if x == '1' else -new_hypercube_length / 2,
                        list(bin(i)[2:].zfill(self.dimension))),
                    dtype=np.float)
                new_hypercubes.append(Hypercube(new_hypercube_length, old_center + center_translation))

            return [UcbNode(self, self.h + 1, new_hypercubes[:int(num_new_hypercubes / 2)]),
                    UcbNode(self, self.h + 1, new_hypercubes[int(num_new_hypercubes / 2):])]
        else:
            return [UcbNode(self, self.h + 1, self.hypercube_list[:int(len(self.hypercube_list) / 2)]),
                    UcbNode(self, self.h + 1, self.hypercube_list[int(len(self.hypercube_list) / 2):])]

    def contains_context(self, context):
        for hypercube in self.hypercube_list:
            if hypercube.is_pt_in_hypercube(context):
                return True
        return False

    def __str__(self):
        return str(self.h) + ": " + str(self.hypercube_list)

    def __repr__(self):
        return self.__str__()
