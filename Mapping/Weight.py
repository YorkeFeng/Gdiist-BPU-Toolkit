"""Network adapter base class
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np

from Common.Common import Fake, div_round_up, get_hex_data


class Weight40nm():
    def __init__(self, connection_matrix, neuron_num, neuron_scale, config) -> None:
        self.connection_matrix = connection_matrix
        self.neuron_num = neuron_num

        if neuron_scale > 1 or neuron_scale < 0:
            raise NotImplementedError(
                "Neuron scale is only supported (0, 1.0]")
        else:
            self.neuron_scale = neuron_scale

        self.fanout = 10
        self.max_neurons_per_npu: int = config['Npu_NeuronNum']
        self.npus_per_chip: int = config['Tile_NpuNum']
        self.neurons_per_chip = config['Npu_NeuronNum'] * config['Tile_NpuNum']
        self.used_chip_num = div_round_up(neuron_num, self.neurons_per_chip)

        self.weight_addr_0: int = 256 * (16 * (self.used_chip_num - 1) + self.npus_per_chip) + \
            16 * (self.max_neurons_per_npu * 2)

    def weight_addr(self):
        fanout = self.fanout
        dst_id_count = []
        results = []

        if self.used_chip_num == 1:
            for dst_ids in self.connection_matrix.values():
                dst_id_count += [len(dst_ids)]
            dst_id_count += [0] * (self.neurons_per_chip -
                                   self.neuron_num)    # 4096 - 1000
            dst_id_cumsum = np.cumsum(
                [self.weight_addr_0] + dst_id_count[:-1])  # 权重矩阵的地址要动态变化
            results.append((dst_id_cumsum << fanout).astype(
                "<u4") + np.array(dst_id_count).astype("<u4"))

        else:
            tmp_file = self.used_chip_num
            dst_id_count = [[] for i in range(tmp_file)]
            for dst_ids in self.connection_matrix.values():
                split_dst = [[] for i in range(tmp_file)]
                for key in dst_ids.keys():
                    split_dst[int(key) // self.neurons_per_chip] += [key]

                for id in range(tmp_file):
                    dst_id_count[id] += [len(split_dst[id])]

            for ids in range(tmp_file - 1):
                dst_id_count[ids] += [0] * \
                    (self.neurons_per_chip * tmp_file - self.neuron_num)
                dst_id_cumsum = np.cumsum(
                    [self.weight_addr_0] + dst_id_count[ids][:-1])
                results.append((dst_id_cumsum << fanout).astype(
                    "<u4") + np.array(dst_id_count[ids]).astype("<u4"))
            dst_id_count[-1] += [0] * \
                (self.neurons_per_chip * tmp_file - self.neuron_num)
            dst_id_cumsum = np.cumsum(
                [self.weight_addr_0] + dst_id_count[-1][:-1])
            results.append((dst_id_cumsum << fanout).astype(
                "<u4") + np.array(dst_id_count[-1]).astype("<u4"))

        return results

    def default_init_value(self):
        # pylint: disable-next=too-many-function-args
        init_value = int(np.single(-60.0).view("uint32").astype("<u4"))
        self.tau_ex = int(
            np.single(np.exp(-1 / 5.0)).view("uint32").astype("<u4"))
        self.tau_ih = int(
            np.single(np.exp(-1 / 10.0)).view("uint32").astype("<u4"))

        if self.used_chip_num == 1:
            tmp_result = [
                [init_value] * self.neuron_num +
                [0] * (16384 - self.neuron_num),
                [10] * self.neuron_num + [0] * (16384 - self.neuron_num),
                [1] * self.neuron_num + [0] * (16384 - self.neuron_num),
                [0] * 16384,
                [0] * 16384,
                [0] * 16384,
                [0] * 16384,
                [0] * 16384]
            result = [np.array(tmp_result).T.reshape(-1,)]

        else:
            tmp_file = self.used_chip_num
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
                [init_value] * (self.neuron_num - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                    16384 - self.neuron_num + (tmp_file-1) * self.neurons_per_chip),
                [10] * (self.neuron_num - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                    16384 - self.neuron_num + (tmp_file-1) * self.neurons_per_chip),
                [1] * (self.neuron_num - (tmp_file-1) * self.neurons_per_chip) + [0] * (
                    16384 - self.neuron_num + (tmp_file-1) * self.neurons_per_chip),
                [0] * 16384,
                [0] * 16384,
                [0] * 16384,
                [0] * 16384,
                [0] * 16384,
            ]
            result = np.array(result).transpose(
                0, 2, 1).reshape(tmp_file, -1)

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

    def dst_weight(self):
        tmp_file = self.used_chip_num
        ex_neuron_num = int(self.neuron_num * self.neuron_scale)
        dst_and_weight = [[] for _ in range(tmp_file)]
        for src_id in range(ex_neuron_num):
            # content of line: [dst_id, weight value, dst_id, weight_value ...]
            # 突触存储的数据类型为：flag， neuron_id, weight_value_1， weight_value_2;在单个突触电流的情况下，flag不起作用，此时默认weight_value_2为0.
            for dst_id, weight_value in self.connection_matrix[src_id].items():
                # 兴奋性神经元的突触权重格式为：flag， neuron_id, weight_value，0
                chip_id = int(dst_id) // self.neurons_per_chip
                dst_id_tmp = dst_id - self.neurons_per_chip*chip_id
                dst_and_weight[chip_id] += [0, weight_value, dst_id_tmp, 0]

        for src_id in range(ex_neuron_num, self.neuron_num):
            for dst_id, weight_value in self.connection_matrix[src_id].items():
                # 抑制性神经元的突触权重格式为：flag， neuron_id, 0， weight_value
                chip_id = int(dst_id) // self.neurons_per_chip
                dst_id_tmp = dst_id - self.neurons_per_chip * chip_id
                dst_and_weight[chip_id] += [weight_value, 0, dst_id_tmp, 0]

        for i in range(tmp_file):
            dst_and_weight[i] = np.array([dst_and_weight[i]])

        return dst_and_weight

    def write_to_bin(self, output_dir):
        output_dir = Path(output_dir) / 'weight'
        output_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
        tmp_file = self.used_chip_num

        # part 1: 神经元索引
        weight_addr = self.weight_addr()

        # part 2: 神经元状态备份，每个神经元8个32bit
        init_value = self.init_value()

        # part 3: 连接+权重，每条128bit
        dst_weight = self.dst_weight()

        for ids in range(tmp_file):
            file_path = output_dir / f"weight_{ids}.bin"
            file_path.unlink(missing_ok=True)
            # index
            if os.path.exists(file_path):
                os.remove(file_path)

            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path,
                        arr=weight_addr[ids], dtype="<u4")
            Fake.fwrite(file_path=file_path,
                        arr=init_value[ids], dtype="<u4")
            Fake.fwrite(file_path=file_path,
                        arr=dst_weight[ids], dtype="<u4")
