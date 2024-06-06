#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yukun Feng
# @Date: 2023-10-16

import re
from collections import defaultdict

import brainpy as bp
import numpy as np
from scipy.sparse import csr_matrix


class BrainpyBase():
    def __init__(self, network) -> None:
        """Analyze the brainpy model
           to get the link matrix and number of neurons.

        Parameters
        ----------
        network : Brainpy model e.g. EInet
        """
        self.network = network

        self.index_map = {}
        last_index = 0
        for name in self.network.nodes().subset(bp.DynamicalSystem):
            if isinstance(self.network.nodes().subset(bp.DynamicalSystem)[name], bp.NeuGroupNS):
                start_index = last_index
                end_index = start_index + \
                    self.network.nodes().subset(
                        bp.DynamicalSystem)[name].size[0]
                self.index_map[name] = (start_index, end_index)
                last_index = end_index

        self.layers = {}
        for name in self.network.nodes().subset(bp.DynamicalSystem):
            if re.match(r"[A-Z]\w+\d+", name):
                self.layers[name] = self.network.nodes().subset(
                    bp.DynamicalSystem)[name]

    def get_connection_matrix(self):
        self.connection_matrix = defaultdict(dict)
        for conn in self.layers.values():
            # 加入
            if isinstance(conn, bp.dyn.SynConn):
                # update connection matrix
                src_offset = self.index_map[conn.pre.name][0]
                dst_offset = self.index_map[conn.post.name][0]
                # indices, indptr = conn.conn_mask
                # indices, indptr = conn.conn.require('csr')
                indices = conn.comm.indices
                indptr = conn.comm.indptr

                rows = indptr.size - 1
                cols = int(np.max(indices) + 1)
                if isinstance(conn.comm.weight, float):
                    data = np.ones(indptr[-1]) * conn.comm.weight
                    compressed = csr_matrix((data, indices, indptr), shape=(
                        rows, cols), dtype=np.float32).tocoo()
                    for i, j, k in zip(compressed.row, compressed.col, compressed.data):
                        src_abs_id = int(i + src_offset)
                        dst_abs_id = int(j + dst_offset)
                        self.connection_matrix[src_abs_id][dst_abs_id] = np.single(
                            k).view("uint32")
                else:
                    data = np.ones(
                        indptr[-1], dtype=np.uint32) * conn.comm.weight
                    compressed = csr_matrix((data, indices, indptr), shape=(
                        rows, cols), dtype=np.uint32).tocoo()
                    for i, j, k in zip(compressed.row, compressed.col, compressed.data):
                        src_abs_id = int(i + src_offset)
                        dst_abs_id = int(j + dst_offset)
                        self.connection_matrix[src_abs_id][dst_abs_id] = np.uint32(
                            k).view("uint32")
        return self.connection_matrix

    def get_neuron_num(self):
        # 返回每一层的神经元的指数范围
        """Neuron layer to neuron start index and end index. e.g.

        ```python
        {"X": (0, 784), "Y": (784, 1409)}
        ```

        or

        ```python
        {"LIF0": (0, 3200), "LIF1": (3200, 4000)}
        ```

        In this example, the total number of neurons is 1409.
        layer with name "X" has 784 neurons and the index of them starts with 0.
        layer with name "Y" has 625 neurons (1409-625). Starts from 784 and ends at 1409 (exclusive).
        The index is 0-based indexing.
        """

        self.neuron_num = max(
            end_index for _, end_index in self.index_map.values())
        return self.neuron_num
