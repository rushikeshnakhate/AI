import pandas as pd
import os
import networkx as nx
import warnings

from blaze.expr import nelements


def get_tr():
    return pd.read_csv(r'D:\AI\week5\data\train.csv')


if __name__ == '__main__':
    graph_str = {
        'A': ['B', 'E'],
        'B': ['A', 'C', 'E'],
        'C': ['D', 'E'],
        'D': ['C', 'E'],
        'E': ['D', 'B', 'A'],
        'F': ['D']
    }

    ans_1 = {
        'A': ['B', 'G', 'H'],
        'B': ['A', 'D', 'C'],
        'C': ['B', 'D', 'E'],
        'E': ['C', 'D', 'F'],
        'F': ['E', 'D', 'G'],
        'G': ['A', 'F', 'D', 'H']
    }

    G = nx.from_dict_of_lists(ans_1)
    print(type(G))
    nx.draw_networkx(G, with_labels=True, node_color='cyan', node_size=500)


class Tree:
    def __init__(self, val, left_child=None, right_child=None):
        self.val = val
        self.left = left_child
        self.right = right_child

    def __str__(self):
        if self.left is None and self.right is None:
            return '%s(%s)' % s(self.__class__.__name__, str(self.val))
        else:
            return '%s(%s,%s,%s)' % (self.__class__.__name__, self.val, self.left, self.right)

    def __repr__(self):
        if self.left is None and self.right is None:
            return '%s(%s)' % (self.__class__, repr(self.val))
        else:
            return '%s(%r,%r,%r)' % (self.__class__.__name__, self.val, self.left, self.right)


class Test:
    def __init__(self):
        print("construcntor")


if __name__ == '__main__':
    X = Test()
