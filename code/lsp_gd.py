# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import numpy as np
from sklearn.neighbors import KDTree
from util import draw_graph_forceatlas2, write_graphml
import networkx as nx

epsilon = 1e-5


def lsp_graph(X, nr_neighbors, metric, labels=None):
    size = len(X)

    # adjusting the number of neighbors in case it is larger than the dataset
    nr_neighbors = min(nr_neighbors, size - 1)

    tree = KDTree(X, leaf_size=2, metric=metric)
    dists, indexes = tree.query(X, k=nr_neighbors + 1)

    # transforming distances to similarities
    alpha = 1.0 / (dists + epsilon)

    # ignoring similarity to itself
    for i in range(size):
        for j in range(nr_neighbors + 1):
            if indexes[i][j] == i:
                alpha[i][j] = 0

    # creating the graph
    g = nx.Graph()

    for i in range(size):
        g.add_node(i)

    # set labels as node attribute
    if labels is not None:
        nx.set_node_attributes(g, dict(enumerate(map(str, labels))), name='label')

    for i in range(size):
        sum_alpha = np.sum(alpha[i])

        for j in range(nr_neighbors + 1):
            if indexes[i][j] != i:
                g.add_edge(i, indexes[i][j], weight=(alpha[i][j] / sum_alpha))
                g.add_edge(indexes[i][j], i, weight=(alpha[i][j] / sum_alpha))

    return g


def gd_lsp(X, labels, filename_fig, filename_graph, nr_neighbors=10):
    metric = 'euclidean'

    g = lsp_graph(X,
                  nr_neighbors=nr_neighbors,
                  metric=metric,
                  labels=labels)

    pos = draw_graph_forceatlas2(X, g, labels, filename_fig)
    write_graphml(g, pos, filename_graph)

