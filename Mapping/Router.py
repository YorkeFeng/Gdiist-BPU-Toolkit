#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yukun Feng
# @Date: 2023-12-10


import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from Common.Common import Fake, array_to_hex, div_round_up, save_hex_file

DIRECTION_NAME = ('WEST', 'SOUTH', 'EAST', 'NORTH')

FIVE_DIRECTION = range(5)


class DIRECTION:
    WEST = 0
    SOUTH = 1
    EAST = 2
    NORTH = 3
    LOCAL = 4


class OneChip:
    def __init__(self, chip_id) -> None:
        self.id = chip_id
        self.edge_count = 0
        self.vertex_count = 0
        self.vertex = []    # chip vertex
        self.edge_map = {}  # edges with the dst vertex in chip
        self.msg_queue = []
        self.msg_queue_next = [[] for _ in range(5)]
        self.neighbor_chips = {}  # {OneChip * 4}
        self.routing_table = {}  # Dict{routing: {src_v: <W, S, E, N, Local>}}
        self.bfs_record = {}


class Router():
    def __init__(self, config, neuron_num) -> None:
        self.X_NumOfChips = config['X_TileNum']
        self.Y_NumOfChips = config['Y_TileNum']
        self.NPUNumOfChip = config['Tile_NpuNum']
        self.NodeNumOfNpu = config['Npu_NeuronNum']
        self.neuron_num = neuron_num

        used_npu = div_round_up(neuron_num, self.NodeNumOfNpu)
        used_chip = div_round_up(used_npu, self.NPUNumOfChip)
        used_y = div_round_up(used_chip, self.X_NumOfChips)
        print(used_y)
        if used_y > self.Y_NumOfChips:
            logger.critical(
                f'Error: {neuron_num} neurons can not load into {self.X_NumOfChips * self.Y_NumOfChips} chips.')
            exit(1)
        else:
            self.Y_NumOfChips = used_y

        if used_y == 1:
            self.X_NumOfChips = used_chip

        # init chip matrix
        self.TotalChips = self.X_NumOfChips * self.Y_NumOfChips
        self.chip_matrix = [OneChip(i) for i in range(self.TotalChips)]
        self.npu_num = self.NPUNumOfChip * self.TotalChips

        # npus into chips
        self.npus_to_chips = []
        for i in range(self.TotalChips):
            self.npus_to_chips.extend([i] * self.NPUNumOfChip)

        # build chip connection
        for i in range(self.TotalChips):
            x, y = self.convert_id_to_xy(i)
            if x > 0:
                self.chip_matrix[i].neighbor_chips[DIRECTION.NORTH] = self.chip_matrix[i -
                                                                                       self.Y_NumOfChips]
            if x < self.X_NumOfChips-1:
                self.chip_matrix[i].neighbor_chips[DIRECTION.SOUTH] = self.chip_matrix[i +
                                                                                       self.Y_NumOfChips]
            if y > 0:
                self.chip_matrix[i].neighbor_chips[DIRECTION.WEST] = self.chip_matrix[i - 1]
            if y < self.Y_NumOfChips - 1:
                self.chip_matrix[i].neighbor_chips[DIRECTION.EAST] = self.chip_matrix[i + 1]

        self.router_load = np.zeros(
            (self.TotalChips, 5), dtype=np.int64)

    def convert_id_to_xy(self, chip_id):
        chip_x = chip_id // self.Y_NumOfChips
        chip_y = chip_id % self.Y_NumOfChips

        return chip_x, chip_y

    def get_one_src_routing(self, src_id, dst_id_list):
        """_summary_

        Parameters
        ----------
        src_id : int
            Routing source chip id
        dst_id_list : list [id_0, id_1, ...]
            Routing target chips id collaction

        """
        routing = np.zeros((self.TotalChips, 5))
        routing_record = [routing]
        path_lens = [-1] * self.TotalChips
        src_x, src_y = self.convert_id_to_xy(src_id)
        dst_set = []
        src_set = [src_id]
        path_lens[src_id] = 0

        # straight forward
        for dst_id in dst_id_list:
            dst_x, dst_y = self.convert_id_to_xy(dst_id)

            # get direction
            diff_x = src_x - dst_x
            diff_y = src_y - dst_y

            if diff_x != 0 and diff_y != 0:
                dst_set.append(dst_id)
                continue

            if diff_x < 0:
                direct = DIRECTION.SOUTH    # South
            elif diff_x > 0:
                direct = DIRECTION.NORTH    # North
            elif diff_y < 0:
                direct = DIRECTION.EAST    # East
            elif diff_y > 0:
                direct = DIRECTION.WEST    # West

            diff = abs(diff_x) + abs(diff_y)
            current_id = src_id
            current_dist = 0
            for i in range(diff):
                routing[current_id, direct] = 1
                current_id = self.chip_matrix[current_id].neighbor_chips[direct].id
                src_set.append(current_id)
                current_dist = current_dist + 1
                path_lens[current_id] = current_dist
            routing[current_id, DIRECTION.LOCAL] = 1
            routing_record.append(routing)

        # logger.info(
        #     f'Finish straight forward, routing_record: \n{np.array(routing)}')

        # non-straight forward
        src_set = list(set(src_set))
        loop_cnt = -1
        while len(dst_set) > 0:
            loop_cnt = loop_cnt + 1
            # logger.info(
            #     f'\n\n===================Non-Straight Forward: Loop {loop_cnt}===================')
            # logger.info(f'src_set: {src_set}')
            # logger.info(f'dst_set: {dst_set}')
            src_cands = []
            dst_cands = []
            dist_cands = []

            # l1_dist
            for dst_id in dst_set:
                dst_x, dst_y = self.convert_id_to_xy(dst_id)
                l1_dist = abs(src_x - dst_x) + abs(src_y - dst_y)

                dist = []
                for src in src_set:
                    x, y = self.convert_id_to_xy(src)
                    dist.append(abs(dst_x-x) + abs(dst_y-y))

                find_flag = False
                while not find_flag:
                    min_dist = min(dist)
                    src_cands_tmp = [s
                                     for i, s in enumerate(src_set) if dist[i] == min_dist]
                    dist_cands_tmp = np.array([
                        path_lens[i] for i in src_cands_tmp]) + min_dist
                    # check is l1_dist
                    bypass_filted_idx = []
                    for i, d in enumerate(dist_cands_tmp):
                        if d == l1_dist or 1:
                            bypass_filted_idx.append(i)
                    if len(bypass_filted_idx) > 0:
                        src_cands.extend([src_cands_tmp[i]
                                         for i in bypass_filted_idx])
                        dst_cands.extend(
                            len([src_cands_tmp[i] for i in bypass_filted_idx]) * [dst_id])
                        dist_cands.extend(
                            len([src_cands_tmp[i] for i in bypass_filted_idx]) * [min_dist])
                        find_flag = True

            # logger.info(f'src_cands = {src_cands}')
            # logger.info(f'dst_id = {dst_cands}')

            # choose min cands
            min_dist = min(dist_cands)
            min_src_dist = self.TotalChips
            for s, d, di in zip(src_cands, dst_cands, dist_cands):
                if di == min_dist and path_lens[s] < min_src_dist:
                    filted_src_cand = s
                    filted_dst_cand = d
                    min_src_dist = path_lens[s]

            # logger.info(
            #     f'Filted src_id = {filted_src_cand}, dst_id = {filted_dst_cand}')

            # search path
            src_cand_x, src_cand_y = self.convert_id_to_xy(filted_src_cand)
            dst_cand_x, dst_cand_y = self.convert_id_to_xy(filted_dst_cand)
            diff_x = src_cand_x - dst_cand_x
            diff_y = src_cand_y - dst_cand_y

            if diff_x <= 0:
                direct_x = DIRECTION.SOUTH    # South
            else:
                direct_x = DIRECTION.NORTH    # North

            if diff_y <= 0:
                direct_y = DIRECTION.EAST    # East
            else:
                direct_y = DIRECTION.WEST    # West

            diff_x = abs(diff_x)
            diff_y = abs(diff_y)

            # current location
            current_id = filted_src_cand
            current_dist = path_lens[current_id]

            while diff_x > 0 or diff_y > 0:
                if (self.router_load[current_id, direct_x] > self.router_load[current_id, direct_y] or diff_x <= 0) and diff_y > 0:
                    diff_y = diff_y - 1
                    routing[current_id, direct_y] = 1
                    current_id = self.chip_matrix[current_id].neighbor_chips[direct_y].id
                    src_set.append(current_id)

                    current_dist = current_dist + 1

                    path_lens[current_id] = current_dist
                else:
                    diff_x = diff_x - 1
                    routing[current_id, direct_x] = 1
                    current_id = self.chip_matrix[current_id].neighbor_chips[direct_x].id
                    src_set.append(current_id)
                    current_dist = current_dist + 1
                    path_lens[current_id] = current_dist
            routing[current_id, DIRECTION.LOCAL] = 1
            routing_record.append(routing)
            if filted_dst_cand == current_id:
                dst_set.remove(current_id)
            else:
                logger.error(
                    f'routing final_id {current_id} is not equal to dst_id {filted_dst_cand}')
                exit(1)
            src_set = list(set(src_set))
            # src_set = sorted(src_set)
            # logger.info(f'src_set: {src_set}')
            # logger.info(f'dst_set: {dst_set}')
            # logger.info(f'path_len: {path_lens}')
        # logger.info(
        #     f'\n\n===================Finish Non-Straight Forward===================')
        # logger.info(f'routing_record: \n{np.array(routing)}')

        # updata router_load
        for chip_id in range(self.TotalChips):
            route_info = routing[chip_id]
            if np.sum(route_info) == 0:
                continue
            for direct in FIVE_DIRECTION:
                self.router_load[chip_id, direct] = self.router_load[chip_id,
                                                                     direct] + routing[chip_id][direct]
        return routing

    def connection_into_npu_flow(self, connection_matrix):
        """_summary_

        Parameters
        ----------
        connection_matrix : dict
            connection_matrix[src_v] = [dst_v1, dst_v2, ...]

        """
        # vertex into npus/chips
        self.vertex_to_npus = []
        self.vertex_to_chips = []
        for i in range(self.neuron_num):
            npu_id = i // self.NodeNumOfNpu
            self.vertex_to_npus.append(npu_id)
            self.vertex_to_chips.append(self.npus_to_chips[npu_id])

        # edge to npu flow
        self.npu_flows_to_chips = {}
        for src_v in connection_matrix:
            for dst_v in connection_matrix[src_v]:
                src_npu = self.vertex_to_npus[src_v]
                dst_chip = self.vertex_to_chips[dst_v]
                if src_npu in self.npu_flows_to_chips:
                    self.npu_flows_to_chips[src_npu].append(dst_chip)
                else:
                    self.npu_flows_to_chips[src_npu] = [dst_chip]

        for src_npu in self.npu_flows_to_chips:
            self.npu_flows_to_chips[src_npu] = list(
                set(self.npu_flows_to_chips[src_npu]))

        print(f'npu_flows_to_chips: {self.npu_flows_to_chips}')

        return self.npu_flows_to_chips

    def gen_routing(self, connection_matrix):
        self.connection_into_npu_flow(connection_matrix)
        route_path = []
        for src_npu in range(self.npu_num):
            if src_npu in self.npu_flows_to_chips:
                src_chip = self.npus_to_chips[src_npu]
                dst_chip = self.npu_flows_to_chips[src_npu]
                routing = self.get_one_src_routing(
                    src_chip, dst_chip)
                route_path.append(routing)
            else:
                route_path.append(np.zeros((self.TotalChips, 5)))
        self.route_path = np.array(route_path)

        return self.route_path


class Router40nm(Router):
    def __init__(self, config, neuron_num) -> None:
        super().__init__(config, neuron_num)
        self.bram_depth = 1024
        self.bram_num = 16

        self.direction_weight = [8, 4, 2, 1, 16]

    def get_bram_index(self, npu_id):
        bram_col = self.bram_num - int(npu_id / self.bram_depth) - 1
        bram_row = npu_id % self.bram_depth
        bram_index = bram_row * self.bram_num + bram_col

        return bram_index

    def write_to_bin(self, save_dir):
        save_dir = Path(save_dir) / 'route_info'
        save_dir.mkdir(parents=True, exist_ok=True)

        # 第一位使能为1
        route_enable = (np.sum(self.route_path, axis=2)
                        > 0).astype("<u2") * (2**15)
        route_tag = np.tile(
            (np.floor(np.mod(np.arange(self.npu_num),
                             (2**20)) / (2**10)) * (2**5)).astype("<u2"),
            (self.TotalChips, 1),
        ).T
        self.route_info = (
            route_enable
            + route_tag
            + self.route_path[:, :, DIRECTION.LOCAL].astype(
                "<u2") * self.direction_weight[DIRECTION.LOCAL]
            + self.route_path[:, :, DIRECTION.WEST].astype(
                "<u2") * self.direction_weight[DIRECTION.WEST]
            + self.route_path[:, :, DIRECTION.SOUTH].astype(
                "<u2") * self.direction_weight[DIRECTION.SOUTH]
            + self.route_path[:, :, DIRECTION.EAST].astype(
                "<u2") * self.direction_weight[DIRECTION.EAST]
            + self.route_path[:, :, DIRECTION.NORTH].astype(
                "<u2") * self.direction_weight[DIRECTION.NORTH]
        )

        for n in range(self.route_info.shape[1]):
            file_path = save_dir / f'route_info_{n}.bin'
            file_path.unlink(missing_ok=True)
            data = np.zeros((self.bram_depth * self.bram_num, 1), "uint16")
            for i in range(self.route_info.shape[0]):
                data[self.get_bram_index(i)] = self.route_info[i, n]
            Fake.fwrite(file_path=file_path, arr=data, dtype="<u2")


class Router28nm(Router):
    def __init__(self, config, neuron_num) -> None:
        super().__init__(config, neuron_num)
        self.bram_depth = 4096
        self.bram_num = 4
        self.bram_block_depth = 1024
        self.bram_block_count = self.bram_num * self.bram_block_depth

        self.direction_weight = [2, 4, 1, 8, 16]

    def get_bram_index(self, npu_id):
        bram_block = int(npu_id / self.bram_block_count)
        block_id = int(npu_id % self.bram_block_count)
        bram_col = self.bram_num - \
            int(block_id / self.bram_block_depth) - 1
        bram_row = block_id % self.bram_block_depth + bram_block * self.bram_block_depth
        bram_index = bram_row * self.bram_num + bram_col

        return bram_index

    def write_to_bin(self, save_dir, file_type='bin'):
        save_dir = Path(save_dir) / 'route_info'
        save_dir.mkdir(parents=True, exist_ok=True)

        # 第一位使能为1
        route_enable = (np.sum(self.route_path, axis=2)
                        > 0).astype("<u2") * (2**15)
        route_tag = np.tile(
            (np.floor(np.mod(np.arange(self.npu_num),
                             (2**20)) / (2**10)) * (2**5)).astype("<u2"),
            (self.TotalChips, 1),
        ).T
        self.route_info = (
            route_enable
            + route_tag
            + self.route_path[:, :, DIRECTION.LOCAL].astype(
                "<u2") * self.direction_weight[DIRECTION.LOCAL]
            + self.route_path[:, :, DIRECTION.WEST].astype(
                "<u2") * self.direction_weight[DIRECTION.WEST]
            + self.route_path[:, :, DIRECTION.SOUTH].astype(
                "<u2") * self.direction_weight[DIRECTION.SOUTH]
            + self.route_path[:, :, DIRECTION.EAST].astype(
                "<u2") * self.direction_weight[DIRECTION.EAST]
            + self.route_path[:, :, DIRECTION.NORTH].astype(
                "<u2") * self.direction_weight[DIRECTION.NORTH]
        )
        for n in range(self.route_info.shape[1]):
            if file_type == 'hex':
                file_path = save_dir / f'route_info_{n}.hex'
            else:
                file_path = save_dir / f'route_info_{n}.bin'
            file_path.unlink(missing_ok=True)
            data = np.zeros((self.bram_depth * self.bram_num, 1), "uint16")
            for i in range(self.route_info.shape[0]):
                data[self.get_bram_index(i)] = self.route_info[i, n]

            if file_type == 'hex':
                hex_file = array_to_hex(data)
                save_hex_file(file_path, hex_file, each_row_num=16)
            else:
                Fake.fwrite(file_path=file_path, arr=data, dtype="<u2")
