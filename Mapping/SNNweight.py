"""Network adapter base class
"""
import os
from dataclasses import dataclass
from functools import cached_property, partial, reduce
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import numpy as np

from Common.Common import Fake, div_round_up, get_hex_data

# from route import Router


def dump_find_weight(connection_item_list, neurons_per_chip, chip_num):
    dst_and_weight = [[] for _ in range(chip_num)]
    for connection_item in connection_item_list:
        for dst_id, weight_value in connection_item:
            chip_id = int(dst_id) // neurons_per_chip
            dst_and_weight[chip_id] += [0, weight_value, dst_id, 0]
    return dst_and_weight


@dataclass
class SNNWeight():
    """Spiking Neural Network with routing and dumping to binary files support."""

    def __init__(self, connection_matrix, neurons, neuronScale, config):
        # NetworkHardwareArchitecture
        """Hardware architecture class with hardware architecture properties."""

        self.max_neurons_per_npu: int = config['Npu_NeuronNum']
        """Max number of neurons per npu. Defaults to `1024`.
        """

        self.npus_per_chip: int = config['Tile_NpuNum']
        """Number of NPUs per chip. Defaults to `16`.
        """

        self.chips_per_board: int = config['TotalTileNum']
        """Number of chips per board. Defaults to `1`.
        Routing will do nothing if only one chip is on board.
        """

        # self.rand_seed: int = config['RandomSeed']
        """Random seed. Defaults to `42`.
        """

        # 20230809补充属性，用于判断神经元类型
        self.scale: float = neuronScale

        # 兴奋性神经元数目
        self.ex_neurons_per_chip: int = 0

        # 抑制性神经元数目
        self.ih_neurons_per_chip: int = 0

        # TODO, uger, 2023.09.27 改成类属性变量
        self.neurons_per_chip: int = config['Npu_NeuronNum'] * \
            config['Tile_NpuNum']

        #
        self._type: str = config['FileType']
        # self.asic: bool = config['IsAsic']

        self.chip_not_full: bool = True if self.neurons_per_chip != 16384 else False

        range_array = np.array((np.arange(self.chips_per_board)))
        to_shape = (int(np.sqrt(self.chips_per_board)),
                    int(np.sqrt(self.chips_per_board)))
        self.chip_shape = np.reshape(range_array, to_shape)

        """Binary output directory. Defaults to `"data_output"`.
        """

        # TODO: change variabler name
        self.chips_per_board = div_round_up(neurons, self.neurons_per_chip)
        # self.npus_per_chip = min(config['NpuNumOfChip'], div_round_up(
        #     neurons, self.neurons_per_chip))

        self.weight_addr_0: int = 256 * (16 * (self.chips_per_board - 1) + self.npus_per_chip) + \
            16 * (self.max_neurons_per_npu * 2)

        """First weight address. Defaults to 0. Can be 4096 or 2304?
        """

        self.connection_matrix = connection_matrix
        # self.layers = layers
        self.neurons = neurons

    """Methods for spiking neural network base class"""

    @cached_property
    def npu2neurons(self) -> Dict[int, List[int]]:
        """NPU ID to neuron IDs map: `{npu_id: [neuron_ids, ...]}`,
        e.g. `{0: [0, 1, ..., 1023], 1: [1024, 1025, ..., 1408]}`

        Returns:
            Dict[int, List[int]]: NPU ID to neurons map.
        """
        result = {}
        offset = 0
        neuron_count = self.neurons
        # neuron_count = 1000
        while neuron_count:
            neuron_count_on_npu = min(self.max_neurons_per_npu, neuron_count)
            result[len(result)] = list(
                range(offset, offset + neuron_count_on_npu))
            offset += neuron_count_on_npu
            neuron_count -= neuron_count_on_npu
        return result

    @property
    def npus(self) -> int:
        """Number of NPUs in the network.

        Returns:
            int: Number of NPUs in the network.
        """
        return len(list(self.npu2neurons))

    @cached_property
    def chip2npu(self) -> Dict[int, List[int]]:
        """Chip id to npu id map: `{chip_id: [npu_id, ...]}`, e.g. `{1: [0, 1, ..., 15]}`
        This is an one to multiple map.

        Returns:
            dict[int, list[int]]: Chip ID to NPU IDs map.
        """
        result = {}
        for npu_id in self.npu2neurons:
            chip_id = int(npu_id / self.npus_per_chip)
            result[chip_id] = result.get(chip_id, [])
            result[chip_id] += [npu_id]
        return result

    @cached_property
    def npu2chip(self) -> Dict[int, int]:
        """NPU ID to chip ID map: `{npu_id: chip_id}`. This is an one to one map.

        Returns:
            dict[int, int]: NPU ID to chip ID map.
        """
        result = {}
        for chip_id, npu_ids in self.chip2npu.items():
            for npu_id in npu_ids:
                result[npu_id] = chip_id

        return result

    @property
    def neuron2chip(self) -> Dict[int, int]:
        """Neuron IDs to chip index map.

        Returns:
            int: Total number or neurons in the network.
        """
        result = {}
        for chip_id, npu_ids in self.chip2npu.items():
            for npu_id in npu_ids:
                for neuron_id in self.npu2neurons[npu_id]:
                    result[neuron_id] = chip_id

        return result

    @cached_property
    def weight_addr(self, fanout=12) -> np.ndarray:
        """Weight value address of each source neuron.

        Returns:
            np.ndarray: Weight value address of each source neuron.
        """
        if self._type == 'bin':
            fanout = 10
        else:
            fanout = 12
        dst_id_count = []
        results = []
        tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
        results = []

        if tmp_file == 1:
            for dst_ids in self.connection_matrix.values():
                dst_id_count += [len(dst_ids)]
            dst_id_count += [0] * (self.neurons_per_chip -
                                   self.neurons)    # 4096 - 1000
            dst_id_cumsum = np.cumsum(
                [self.weight_addr_0] + dst_id_count[:-1])  # 权重矩阵的地址要动态变化
            results = (dst_id_cumsum << fanout).astype(
                "<u4") + np.array(dst_id_count).astype("<u4")

        else:
            dst_id_count = [[] for i in range(tmp_file)]
            for dst_ids in self.connection_matrix.values():
                split_dst = [[] for i in range(tmp_file)]
                for key in dst_ids.keys():
                    split_dst[int(key) // self.neurons_per_chip] += [key]

                for id in range(tmp_file):
                    dst_id_count[id] += [len(split_dst[id])]
                    # dst_id_count[tmp_file - 1] += [len(dst_ids) % self.neurons_per_chip]

            # insert zero
            if self.chip_not_full:
                for fill_col in range(tmp_file - 1):
                    for ids in range(tmp_file):
                        dst_id_count[ids] = (dst_id_count[ids][:fill_col*16384+self.neurons_per_chip] + [0] * (16384 - self.neurons_per_chip) +
                                             dst_id_count[ids][fill_col*16384+self.neurons_per_chip:])

            for ids in range(tmp_file - 1):
                dst_id_count[ids] += [0] * \
                    (self.neurons_per_chip * tmp_file - self.neurons)
                dst_id_cumsum = np.cumsum(
                    [self.weight_addr_0] + dst_id_count[ids][:-1])
                results.append((dst_id_cumsum << fanout).astype(
                    "<u4") + np.array(dst_id_count[ids]).astype("<u4"))
            dst_id_count[-1] += [0] * \
                (self.neurons_per_chip * tmp_file - self.neurons)
            dst_id_cumsum = np.cumsum(
                [self.weight_addr_0] + dst_id_count[-1][:-1])
            results.append((dst_id_cumsum << fanout).astype(
                "<u4") + np.array(dst_id_count[-1]).astype("<u4"))
        return results

    @cached_property
    def default_init_value(self) -> np.ndarray:
        """default initial value of each neuron.

        Returns:
            List[List[int]]: Initial value of each neuron.
        """
        # pylint: disable-next=too-many-function-args
        init_value = int(np.single(-60.0).view("uint32").astype("<u4"))
        self.tau_ex = int(
            np.single(np.exp(-1 / 5.0)).view("uint32").astype("<u4"))
        self.tau_ih = int(
            np.single(np.exp(-1 / 10.0)).view("uint32").astype("<u4"))

        print("self.ex_neurons_per_chip", self.ex_neurons_per_chip)

        # 两种神经元类型
        if self._type == "bin":

            if self.neurons <= self.neurons_per_chip:

                tmp_result = [
                    [init_value] * self.neurons + [0] * (16384 - self.neurons),
                    [10] * self.neurons + [0] * (16384 - self.neurons),
                    [1] * self.neurons + [0] * (16384 - self.neurons),
                    [0] * 16384,
                    [0] * 16384,
                    [0] * 16384,
                    [0] * 16384,
                    [0] * 16384]
                result = np.array(tmp_result).T.reshape(-1,)

            else:
                tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
                result = [[] for i in range(tmp_file)]
                for ids in range(tmp_file-1):
                    result[ids] = [
                        [init_value] * self.neurons_per_chip +
                        [0] * (16384 - self.neurons_per_chip),
                        [10] * self.neurons_per_chip + [0] *
                        (16384 - self.neurons_per_chip),
                        [1] * self.neurons_per_chip + [0] *
                        (16384 - self.neurons_per_chip),
                        [0] * 16384,
                        [0] * 16384,
                        [0] * 16384,
                        [0] * 16384,
                        [0] * 16384]
                result[-1] = [
                    [init_value] * (self.neurons - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                        16384 - self.neurons + (tmp_file-1) * self.neurons_per_chip),
                    [10] * (self.neurons - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                        16384 - self.neurons + (tmp_file-1) * self.neurons_per_chip),
                    [1] * (self.neurons - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                        16384 - self.neurons + (tmp_file-1) * self.neurons_per_chip),
                    [0] * 16384,
                    [0] * 16384,
                    [0] * 16384,
                    [0] * 16384,
                    [0] * 16384,
                ]
                result = np.array(result).transpose(
                    0, 2, 1).reshape(tmp_file, -1)

        elif self._type == "hex":
            if self.neurons <= self.neurons_per_chip:
                result = [
                    [1] * self.neurons + [0] *
                    (self.neurons_per_chip - self.neurons),
                    [1] * self.neurons + [0] *
                    (self.neurons_per_chip - self.neurons),
                    [0] * self.neurons_per_chip,
                    [0] * self.neurons_per_chip,


                    [10] * self.neurons + [0] *
                    (self.neurons_per_chip - self.neurons),

                    [init_value] * self.neurons + [0] *
                    (self.neurons_per_chip - self.neurons),
                    [0] * self.neurons_per_chip,
                    [0] * self.neurons_per_chip,
                    [0] * self.neurons_per_chip,
                    [0] * self.neurons_per_chip,
                    [0] * self.neurons_per_chip,
                ]
            else:
                tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
                result = [[] for i in range(tmp_file)]
                for ids in range(tmp_file - 1):
                    result[ids] = [
                        [1] * self.neurons_per_chip,
                        [1] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,


                        [10] * self.neurons_per_chip,

                        [init_value] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                    ]
                    result[-1] = [
                        [1] * (self.neurons - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                            self.neurons_per_chip - self.neurons + (tmp_file-1) * self.neurons_per_chip),
                        [1] * (self.neurons - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                            self.neurons_per_chip - self.neurons + (tmp_file-1) * self.neurons_per_chip),
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [10] * (self.neurons - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                            self.neurons_per_chip - self.neurons + (tmp_file-1) * self.neurons_per_chip),

                        [init_value] * (self.neurons - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                            self.neurons_per_chip - self.neurons + (tmp_file-1) * self.neurons_per_chip),
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                        [0] * self.neurons_per_chip,
                    ]

        # result = np.array(result)

        # 8 * 1024  num_neurons_per_chip = 1024, neurons_variable_num = 8
        return result

    _initial_value: Optional[np.ndarray] = None
    """Initial value of each neuron. If not set, `self.default_init_value` will be used.
    """

    @property
    def init_value(self) -> np.ndarray:
        """Initial value of each neuron.

        Returns:
            np.ndarray: Initial value of each neuron.
        """
        if self._initial_value is None:
            return self.default_init_value
        return self._initial_value

    @init_value.setter
    def init_value(self, value: np.ndarray) -> None:
        """Set initial value of each neuron.

        Args:
            value (np.ndarray): Initial value.
        """
        self._initial_value = value

    @cached_property
    def property_data(self) -> np.ndarray:
        """Property data to write.

        Returns:
            np.ndarray: Property data to write.
        """

        # result数组需要有26个32bit的数据，与m脚本里的property要对应

        property_reshape = np.array(
            [[1, -52, -60, 6, -50, -60, 1 / 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
        # pylint: disable-next=too-many-function-args
        property_float = np.array(np.single(property_reshape).view("uint32"))
        result = np.zeros((26, 1), dtype="<u4")

        result[0] = np.array([0])
        result[1] = np.array([4096])
        result[2:20] = property_float
        result[20: 27] = np.array([[0, 0, 5000000, 0, 999, 0]]).T

        return result

    def dump(self, output_dir, job=1) -> None:
        """Write binary files and route info to file system."""
        output_dir = Path(output_dir)
        output_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

        self.ex_neurons_per_chip = round(self.neurons * self.scale)
        self.ih_neurons_per_chip = self.neurons - self.ex_neurons_per_chip
        # TODO：
        # dump route info
        # if self.neurons <= self.neurons_per_chip:
        #     for chip_id in range(self.route_info.shape[1]):
        #         file_path = output_dir / 'route_info' / f"route_info_{chip_id}.bin"
        #         file_path.unlink(missing_ok=True)
        #         data = np.zeros((1024 * 16, 1), "uint16")
        #         ss = np.zeros((1024 * 16, 1), "uint16")
        #         tem = np.array([self.route_info[:, np.mod(chip_id, 16)]]).T
        #         ss[: self.route_info.shape[0]] = tem
        #         # print("data_before: ", ss)
        #         for j in range(self.route_info.shape[0]):
        #             # data[j* 16 : self.route_info.shape[0]] = tem[j]
        #             data[j * 16 + 15] = tem[j]
        #         # print("data_after: ", data)
        #         # print(np.equal(data, ss))
        #         Fake.fwrite(file_path=file_path, arr=data, dtype="<u2")

        # dump destination neuron IDs and weight values
        tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
        file_path = Path(output_dir) / 'weight'
        file_path.mkdir(parents=True, exist_ok=True)
        if self._type == 'bin':
            if self.neurons > self.neurons_per_chip:
                for ids in range(tmp_file):
                    file_path = output_dir / 'weight' / f"weight_{ids}.bin"
                    file_path.unlink(missing_ok=True)
                    # index
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    file_path.unlink(missing_ok=True)
                    Fake.fwrite(file_path=file_path,
                                arr=self.weight_addr[ids], dtype="<u4")
                    # ncu
                    Fake.fwrite(file_path=file_path,
                                arr=self.init_value[ids], dtype="<u4")
            else:
                file_path = output_dir / 'weight' / f"weight_0.bin"
                file_path.unlink(missing_ok=True)
                # index
                if os.path.exists(file_path):
                    os.remove(file_path)
                file_path.unlink(missing_ok=True)
                Fake.fwrite(file_path=file_path,
                            arr=self.weight_addr, dtype="<u4")
                # ncu
                Fake.fwrite(file_path=file_path,
                            arr=self.init_value, dtype="<u4")
                # 突触连接的存储处理
        print("Saving Weight...............")
        if self._type == 'bin':
            if self.scale > 1.0:
                raise NotImplementedError("Scale > 1.0 is not supported yet.")

            elif self.scale < 1.0:
                if self.neurons > self.neurons_per_chip:
                    tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
                    file_path_list = []
                    for i in range(tmp_file):
                        # file_path = output_dir / f'weight{i}_temp.bin'
                        file_path = output_dir / 'weight' / f"weight_{i}.bin"
                        file_path_list.append(file_path)
                        # if os.path.exists(file_path):
                        #     os.remove(file_path)

                    fwrite_count = 0
                    fwrite_max = self.neurons

                    for src_id in range(int(self.neurons * self.scale)):
                        # content of line: [dst_id, weight value, dst_id, weight_value ...]
                        # 突触存储的数据类型为：flag， neuron_id, weight_value_1， weight_value_2;在单个突触电流的情况下，flag不起作用，此时默认weight_value_2为0.
                        if fwrite_count == 0:
                            dst_and_weight = [[] for i in range(tmp_file)]
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            # 兴奋性神经元的突触权重格式为：flag， neuron_id, weight_value，0
                            chip_id = int(dst_id) // self.neurons_per_chip
                            dst_id_tmp = dst_id - self.neurons_per_chip*chip_id
                            dst_id_tmp = dst_id_tmp + 16384*chip_id if self.chip_not_full else dst_id_tmp

                            dst_and_weight[chip_id] += [0,
                                                        weight_value, dst_id_tmp, 0]
                            # dst_and_weight[chip_id] += [weight_value]
                            # dst_and_weight[chip_id] += [dst_id_tmp]
                            # dst_and_weight[chip_id] += [0]

                        if fwrite_count < fwrite_max:
                            fwrite_count += 1
                            continue
                        else:
                            fwrite_count = 0
                            for i in range(tmp_file):
                                dst_and_weight[i] = np.array(
                                    [dst_and_weight[i]])
                                Fake.fwrite(
                                    file_path=file_path_list[i], arr=dst_and_weight[i], dtype="<u4")
                    if fwrite_count > 0:
                        for i in range(tmp_file):
                            dst_and_weight[i] = np.array(
                                [dst_and_weight[i]])
                            Fake.fwrite(
                                file_path=file_path_list[i], arr=dst_and_weight[i], dtype="<u4")

                    fwrite_count = 0
                    for src_id in range(int(self.neurons * self.scale), self.neurons):
                        if fwrite_count == 0:
                            dst_and_weight = [[] for i in range(tmp_file)]
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            # 抑制性神经元的突触权重格式为：flag， neuron_id, 0， weight_value
                            chip_id = int(dst_id) // self.neurons_per_chip
                            dst_id_tmp = dst_id - self.neurons_per_chip * chip_id
                            dst_id_tmp = dst_id_tmp + 16384 * chip_id if self.chip_not_full else dst_id_tmp

                            dst_and_weight[chip_id] += [weight_value,
                                                        0, dst_id_tmp, 0]
                            # dst_and_weight[chip_id] += [0]
                            # dst_and_weight[chip_id] += [dst_id_tmp]
                            # dst_and_weight[chip_id] += [0]
                        if fwrite_count < fwrite_max:
                            fwrite_count += 1
                            continue
                        else:
                            fwrite_count = 0
                        for i in range(tmp_file):
                            dst_and_weight[i] = np.array([dst_and_weight[i]])
                            Fake.fwrite(
                                file_path=file_path_list[i], arr=dst_and_weight[i], dtype="<u4")
                    if fwrite_count > 0:
                        for i in range(tmp_file):
                            dst_and_weight[i] = np.array(
                                [dst_and_weight[i]])
                            Fake.fwrite(
                                file_path=file_path_list[i], arr=dst_and_weight[i], dtype="<u4")

                else:
                    for src_id in range(int(self.neurons * self.scale)):
                        # content of line: [dst_id, weight value, dst_id, weight_value ...]
                        # 突触存储的数据类型为：flag， neuron_id, weight_value_1， weight_value_2;在单个突触电流的情况下，flag不起作用，此时默认weight_value_2为0.
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            # weight_value = Fake.typecast(Fake.single(weight_value), "<u4")
                            # dst_and_weight += [dst_id << 32]
                            # dst_and_weight += [weight_value << 32]

                            # 兴奋性神经元的突触权重格式为：flag， neuron_id, weight_value，0
                            dst_and_weight += [0]
                            dst_and_weight += [weight_value]
                            dst_and_weight += [dst_id]
                            dst_and_weight += [0]

                        dst_and_weight = np.array([dst_and_weight])
                        Fake.fwrite(file_path=file_path,
                                    arr=dst_and_weight, dtype="<u4")
                    for src_id in range(int(self.neurons * self.scale), self.neurons):
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            # weight_value = Fake.typecast(Fake.single(weight_value), "<u4")
                            # dst_and_weight += [dst_id << 32]
                            # dst_and_weight += [weight_value << 32]

                            # 抑制性神经元的突触权重格式为：flag， neuron_id, 0， weight_value
                            dst_and_weight += [weight_value]
                            dst_and_weight += [0]
                            dst_and_weight += [dst_id]
                            dst_and_weight += [0]

                        dst_and_weight = np.array([dst_and_weight])
                        Fake.fwrite(file_path=file_path,
                                    arr=dst_and_weight, dtype="<u4")

            else:
                if self.neurons <= self.neurons_per_chip:
                    for src_id in range(self.neurons):
                        # content of line: [dst_id, weight value, dst_id, weight_value ...]
                        # 突触存储的数据类型为：flag， neuron_id, weight_value_1， weight_value_2;在单个突触电流的情况下，flag不起作用，此时默认weight_value_2为0.
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            # weight_value = Fake.typecast(Fake.single(weight_value), "<u4")
                            # dst_and_weight += [dst_id << 32]
                            # dst_and_weight += [weight_value << 32]
                            dst_and_weight += [0]
                            dst_and_weight += [weight_value]
                            dst_and_weight += [dst_id]
                            dst_and_weight += [0]

                        dst_and_weight = np.array([dst_and_weight])
                        Fake.fwrite(file_path=file_path,
                                    arr=dst_and_weight, dtype="<u4")
                else:
                    #####################################################################################
                    t1 = time()
                    tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
                    file_path_list = []
                    for i in range(tmp_file):
                        # file_path = output_dir / f'weight{i}_temp.bin'
                        file_path = output_dir / 'weight' / f"weight_{i}.bin"
                        file_path_list.append(file_path)
                        # if os.path.exists(file_path):
                        #     os.remove(file_path)
                    # TODO, uger, debug, 2023.10.7, 对于没有连接的输出神经元，这里需要更改
                    # for src_id in range(self.neurons):
                    fwrite_count = 0

                    if job > 1:
                        count_injob = 3000
                        count_thresh = count_injob * job
                        src_id_list = list(self.connection_matrix.keys())
                        src_num = len(src_id_list)
                        chunk_size = src_num // count_thresh if src_num % count_thresh == 0 else src_num // count_thresh + 1
                        par_func = partial(
                            dump_find_weight, neurons_per_chip=self.neurons_per_chip, chip_num=tmp_file)
                        for chunk in range(chunk_size):
                            dst_and_weight = [[] for i in range(tmp_file)]
                            data_l = []
                            for i in range(job):
                                data_l.append([list(self.connection_matrix[src_id].items(
                                )) for src_id in src_id_list[chunk * count_thresh:chunk * count_thresh + (i+1) * count_injob]])
                            with Pool(processes=job) as pool:
                                res = pool.map(par_func, data_l)
                                pool.close()
                                pool.join()
                            # # with Pool(processes=job) as pool:
                            # #     res = pool.map(par_func, [list(self.connection_matrix[src_id].items()) for src_id in src_id_list[chunk * count_thresh:(chunk+1) * count_thresh]])
                            # #     pool.close()
                            # #     pool.join()

                            # for r in res:
                            #     for i in range(tmp_file):
                            #         dst_and_weight[i].extend(r[i])
                            # for i in range(tmp_file):
                            #     dst_and_weight[i] = np.array([dst_and_weight[i]])
                            #     Fake.fwrite(file_path=file_path_list[i], arr=dst_and_weight[i], dtype="<u4")
                    else:
                        for src_id in self.connection_matrix.keys():
                            if fwrite_count == 0:
                                dst_and_weight = [[] for i in range(tmp_file)]
                            # content of line: [dst_id, weight value, dst_id, weight_value ...]
                            # 突触存储的数据类型为：flag， neuron_id, weight_value_1， weight_value_2;在单个突触电流的情况下，flag不起作用，此时默认weight_value_2为0.
                            for dst_id, weight_value in self.connection_matrix[src_id].items():
                                chip_id = int(dst_id) // self.neurons_per_chip
                                dst_and_weight[chip_id] += [0,
                                                            weight_value, dst_id, 0]
                            if fwrite_count < 1000:
                                fwrite_count = fwrite_count + 1
                                continue
                            else:
                                fwrite_count = 0
                            for i in range(tmp_file):
                                dst_and_weight[i] = np.array(
                                    [dst_and_weight[i]])
                                Fake.fwrite(
                                    file_path=file_path_list[i], arr=dst_and_weight[i], dtype="<u4")
                        if fwrite_count > 0:
                            for i in range(tmp_file):
                                dst_and_weight[i] = np.array(
                                    [dst_and_weight[i]])
                                Fake.fwrite(
                                    file_path=file_path_list[i], arr=dst_and_weight[i], dtype="<u4")
                    t2 = time()
                    print(f'time cost = {t2-t1}s.')
                    #####################################################################################

        elif self._type == 'hex':
            if self.neurons <= self.neurons_per_chip:
                weight_path = output_dir / 'weight_base1_float32_2K_LIF.hex'
                if os.path.exists(weight_path):
                    os.remove(weight_path)
                if self.scale > 1.0:
                    raise NotImplementedError(
                        "Scale > 1.0 is not supported yet.")

                elif self.scale < 1.0:
                    for src_id in range(int(self.neurons * self.scale)):
                        # content of line: [dst_id, weight value, dst_id, weight_value ...]
                        # 突触存储的数据类型为：flag， neuron_id, weight_value_1， weight_value_2;在单个突触电流的情况下，flag不起作用，此时默认weight_value_2为0.
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            dst_and_weight = []

                            # 新的权重矩阵调整为256bit，为原有weight长度的2倍
                            dst_and_weight += [dst_id]
                            dst_and_weight += [0]
                            dst_and_weight += [weight_value]
                            dst_and_weight += [0]

                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight_new = np.array(dst_and_weight)
                            dst_and_weight_new = list(
                                map(lambda x: get_hex_data(hex(x)),  dst_and_weight_new))
                            dst_and_weight_new = reduce(
                                lambda x, y: x + y, dst_and_weight_new)
                            with open(weight_path, 'a') as f_in:
                                f_in.write(dst_and_weight_new + '\n')

                    for src_id in range(int(self.neurons * self.scale), self.neurons):
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            dst_and_weight = []
                            dst_and_weight += [dst_id]
                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [weight_value]

                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight_new = np.array(dst_and_weight)
                            dst_and_weight_new = list(
                                map(lambda x: get_hex_data(hex(x)),  dst_and_weight_new))
                            dst_and_weight_new = reduce(
                                lambda x, y: x + y, dst_and_weight_new)
                            with open(weight_path, 'a') as f_in:
                                f_in.write(dst_and_weight_new + '\n')

                else:
                    for src_id in range(self.neurons):
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            dst_and_weight = []
                            dst_and_weight += [dst_id]
                            dst_and_weight += [0]
                            dst_and_weight += [weight_value]
                            dst_and_weight += [0]

                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight_new = np.array(dst_and_weight)
                            dst_and_weight_new = list(
                                map(lambda x: get_hex_data(hex(x)),  dst_and_weight_new))
                            dst_and_weight_new = reduce(
                                lambda x, y: x + y, dst_and_weight_new)

                            with open(weight_path, 'a') as f_in:
                                f_in.write(dst_and_weight_new + '\n')

                current_num = (self.neurons) ** 2 * 0.02
                dst_and_weight_new = '0' * 64
                for i in range(int(20480 - current_num)):
                    with open(weight_path, 'a') as f_in:
                        f_in.write(dst_and_weight_new + '\n')
            else:
                tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
                weight_path_list = []
                for i in range(tmp_file):
                    weight_path = output_dir / \
                        f'weight{i}_base1_float32_2K_LIF.hex'
                    weight_path_list.append(weight_path)
                    if os.path.exists(weight_path):
                        os.remove(weight_path)
                if self.scale > 1.0:
                    raise NotImplementedError(
                        "Scale > 1.0 is not supported yet.")

                elif self.scale < 1.0:
                    for src_id in range(int(self.neurons * self.scale)):
                        # content of line: [dst_id, weight value, dst_id, weight_value ...]
                        # 突触存储的数据类型为：flag， neuron_id, weight_value_1， weight_value_2;在单个突触电流的情况下，flag不起作用，此时默认weight_value_2为0.
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            dst_and_weight = []

                            # 新的权重矩阵调整为256bit，为原有weight长度的2倍
                            dst_and_weight += [dst_id]
                            dst_and_weight += [weight_value]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight_new = np.array(dst_and_weight)
                            dst_and_weight_new = list(
                                map(lambda x: get_hex_data(hex(x)),  dst_and_weight_new))
                            dst_and_weight_new = reduce(
                                lambda x, y: x + y, dst_and_weight_new)
                            with open(weight_path, 'a') as f_in:
                                f_in.write(dst_and_weight_new + '\n')

                    for src_id in range(int(self.neurons * self.scale), self.neurons):
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            dst_and_weight = []
                            dst_and_weight += [dst_id]
                            dst_and_weight += [0]
                            dst_and_weight += [weight_value]
                            dst_and_weight += [0]

                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight_new = np.array(dst_and_weight)
                            dst_and_weight_new = list(
                                map(lambda x: get_hex_data(hex(x)),  dst_and_weight_new))
                            dst_and_weight_new = reduce(
                                lambda x, y: x + y, dst_and_weight_new)
                            with open(weight_path, 'a') as f_in:
                                f_in.write(dst_and_weight_new + '\n')

                else:

                    input_lens = self.neurons
                    print("Begin saving weight...............")
                    for src_id in range(input_lens):
                        dst_and_weight = []
                        for dst_id, weight_value in self.connection_matrix[src_id].items():
                            chip_id = int(dst_id) // self.neurons_per_chip

                            dst_and_weight = []
                            dst_and_weight += [dst_id]
                            dst_and_weight += [weight_value]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]
                            dst_and_weight += [0]

                            dst_and_weight_new = np.array(dst_and_weight)
                            dst_and_weight_new = list(
                                map(lambda x: get_hex_data(hex(x)),  dst_and_weight_new))
                            dst_and_weight_new = reduce(
                                lambda x, y: x + y, dst_and_weight_new)

                            weight_path = weight_path_list[chip_id]
                            with open(weight_path, 'a') as f_in:
                                f_in.write(dst_and_weight_new + '\n')

        # dump weight address and initial value
        print("Saving Index...............")
        if self._type == 'bin':
            if type(self.weight_addr) == tuple:
                for ids in range(tmp_file):
                    file_path = output_dir / 'weight' / f"index_{ids}.bin"

                    if os.path.exists(file_path):
                        os.remove(file_path)

                    file_path.unlink(missing_ok=True)
                    Fake.fwrite(file_path=file_path,
                                arr=self.weight_addr[ids], dtype="<u4")
            # else:
            #     file_path = output_dir / 'weight' / "weight_0.bin"
            #     file_path.unlink(missing_ok=True)
            #     Fake.fwrite(file_path=file_path, arr=self.weight_addr, dtype="<u4")

        elif self._type == 'hex':
            tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
            if tmp_file == 1:
                index_path = output_dir / "index_base1_float32_1K_LIF.hex"
                if os.path.exists(index_path):
                    os.remove(index_path)
                temp = self.weight_addr
                temp = temp.reshape(-1, 8)  # (512, 8)

                # 每行8个地址，每个地址为32bit，每个地址为8个16进制数，每个数字占4bit，一个字节
                # 硬件能够自动换行
                for i in range(temp.shape[0]):
                    data_tmp = list(
                        map(lambda x: get_hex_data(hex(x)), temp[i][:]))
                    data_tmp = reduce(lambda x, y: x + y, reversed(data_tmp))
                    with open(index_path, 'a') as f_in:
                        f_in.write(data_tmp + '\n')
            else:
                for ids in range(tmp_file):
                    index_path = output_dir / \
                        "index{ids}_base1_float32_1K_LIF.hex"
                    if os.path.exists(index_path):
                        os.remove(index_path)
                    temp = self.weight_addr[ids]
                    temp = temp.reshape(-1, 8)
                    for i in range(temp.shape[0]):
                        data_tmp = list(
                            map(lambda x: get_hex_data(hex(x)), temp[i][:]))
                        data_tmp = reduce(lambda x, y: x + y, data_tmp)
                        with open(index_path, 'a') as f_in:
                            f_in.write(data_tmp + '\n')

        print("Saving NCU...............")
        if self._type == 'bin':
            if type(self.init_value) == tuple:

                for ids in range(len(self.init_value)):

                    file_path = output_dir / 'weight' / f"ncu_temp{ids}.bin"
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    for i in range(np.array(self.init_value[ids]).shape[1]):
                        for j in range(np.array(self.init_value[ids]).shape[0]):
                            Fake.fwrite(file_path=file_path, arr=np.array(
                                self.init_value[ids])[j][i], dtype="<u4")  # 小端模式，最大为4bit
            # else:
            #     for i in range(self.neurons_per_chip):
            #         for j in range(self.init_value.shape[0]):
            #             Fake.fwrite(file_path=file_path, arr=self.init_value[j][i], dtype="<u4")  # 小端模式，最大为4bit

        elif self._type == 'hex':
            ncu_path = output_dir / 'ncu_base1_float32_2K_LIF.hex'
            if os.path.exists(ncu_path):
                os.remove(ncu_path)
            if self.neurons <= self.neurons_per_chip:
                # (4096, 8)，每个神经元8个变量，总共4096个神经元
                temp = np.array(self.init_value[:][:]).T
                # for i in range(0, temp.shape[0]):
                #         # 转置之后每次存储一行数据并拼接为字符串
                #         data_tmp = list(map(lambda x : get_hex_data(hex(x)), temp[:][i]))
                #         data_tmp = reduce(lambda x, y : x + y, data_tmp)
                #         with open(ncu_path, 'a') as f_in:
                #             f_in.write(data_tmp + '\n')

                for i in range(0, temp.shape[0]):
                    # 转置之后每次存储一行数据并拼接为字符串
                    data_tmp = list(map(lambda x: get_hex_data(
                        hex(int(x * 2 ** 7)), lens=2), temp[i][:4]))
                    data_tmp = reduce(lambda x, y: x + y, data_tmp)
                    with open(ncu_path, 'a') as f_in:
                        f_in.write(data_tmp)

                    data_tmp = list(map(lambda x: get_hex_data(
                        hex(x), reverse=True), temp[i][4:]))
                    data_tmp = reduce(lambda x, y: x + y, data_tmp)
                    with open(ncu_path, 'a') as f_in:
                        f_in.write(data_tmp + '\n')
            else:
                tmp_file = self.neurons // self.neurons_per_chip if self.neurons % self.neurons_per_chip == 0 else self.neurons // self.neurons_per_chip + 1
                for i in range(tmp_file):
                    temp = np.array(self.init_value[i]).T
                    for i in range(0, temp.shape[0]):
                        # 转置之后每次存储一行数据并拼接为字符串
                        data_tmp = list(map(lambda x: get_hex_data(
                            hex(int(x * 2 ** 7)), lens=2), temp[i][:4]))
                        data_tmp = reduce(lambda x, y: x + y, data_tmp)
                        with open(ncu_path, 'a') as f_in:
                            f_in.write(data_tmp)

                        data_tmp = list(map(lambda x: get_hex_data(
                            hex(x), reverse=True), temp[i][4:]))
                        data_tmp = reduce(lambda x, y: x + y, data_tmp)
                        with open(ncu_path, 'a') as f_in:
                            f_in.write(data_tmp + '\n')

        else:
            raise NotImplementedError("Type is not supported yet.")

        v = False
        if v:
            v_out_ref_path = output_dir / 'v_out_ref_hex' / \
                'v_out_ref_base1_float32_1K_LIF.hex'
            if os.path.exists(v_out_ref_path):
                os.remove(v_out_ref_path)
            v_name = "v_out_ref"
            v = np.load(
                r"D:\toMW\gdiist-flow-py-master_0911\HD_data\soft_data\N_V.npy")

            def expand(x):
                lens = 4096 - len(x)
                x = np.append(x, np.zeros(lens))
                return x
            temp = (list(map(lambda x: expand(x), v)))    # 100 4096
            for i in range(len(temp)):
                for j in range(len(temp[0])):
                    data_tmp = list(map(lambda x: get_hex_data(
                        hex(np.single(x).view("uint32"))), [temp[i][j]]))
                    data_tmp = reduce(lambda x, y: x + y, data_tmp)
                    with open(v_out_ref_path, 'a') as f_in:
                        f_in.write(data_tmp + '\n')
                v_out_ref_path = output_dir / \
                    'v_out_ref_hex' / (v_name + str(i) + '.hex')
                if os.path.exists(v_out_ref_path):
                    os.remove(v_out_ref_path)
        if not hex:
            tmp_weight_data = np.array(
                [np.fromfile(output_dir / 'weight' / "tmp.bin", dtype="<u8")]).T
            Fake.fwrite(file_path=file_path, arr=tmp_weight_data, dtype="<u8")
