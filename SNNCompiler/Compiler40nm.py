#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yukun Feng
# @Date: 2023-10-23

from pathlib import Path

import numpy as np

from Common.Common import Fake, div_round_up

from .func_2_smt import SMT96_Aisc_Base, func_to_32bit_smt


class Compiler40nmSNN():
    def __init__(self, config, neuron_num, net) -> None:
        self.npu_neuron_num = config['Npu_NeuronNum']
        self.chip_npu_num = config['Tile_NpuNum']
        self.npu_num = div_round_up(neuron_num, self.npu_neuron_num)
        self.chip_num = div_round_up(self.npu_num, self.chip_npu_num)
        print(f'chip num = {self.chip_num}, npu_num = {self.npu_num}')
        self.ndma_staddr = 256 * (16 * (self.chip_num-1) + self.chip_npu_num)

        self.neuron_num = neuron_num
        self.net = net
        self.config = config
        self.asic_flag = self.config['IsAsic']
        self.config['neuron_num'] = self.neuron_num

        self.npu_on_s = []
        self.npu_after_s = []
        self.npu_all0_s = []
        self.npu_spike_s = []
        self.spdma_thres = int('000A_0010', 16)
        for j in range(self.chip_num):
            if self.npu_num - self.chip_npu_num * (j) > self.chip_npu_num:
                tmp_s = '1' * self.chip_npu_num
            else:
                tmp_s = '1' * (self.npu_num - self.chip_npu_num * j) + '0' * \
                    (self.chip_npu_num - (self.npu_num - self.chip_npu_num * j))

            value = 0
            for i in range(self.chip_npu_num):
                value = value + int(tmp_s[i]) * (2 ** (self.chip_npu_num-1-i))
            value = value << 16
            self.npu_on_s.append(value + 572)  # 5: 556 7: 572
            self.npu_after_s.append(value + 63)  # 5: 47 7: 63
            self.npu_all0_s.append(value + 60)  # 5: 44 7: 60
            self.npu_spike_s.append(value + 62)  # 5: 46 7: 62

        if self.asic_flag:
            self.smt96_compiler = SMT96_Aisc_Base(self.net, self.config)

    def get_smt_32bit_result(self):
        self.smt_result, self.all_constants, config = func_to_32bit_smt(
            self.net, self.config)

        v_reset = config['V_reset']
        t_refrac = config['T_refrac']
        v_thresh = config['V_thresh']
        v_rest = config['V_reset']
        RC_decay = float(config['RC_decay'])
        tw = config['tw']
        step_max = config['step_max']
        neu_nums = config['neu_num']
        rate = config['rate']

        shared_property = [self.spdma_thres, v_reset, t_refrac,
                           v_thresh, v_rest] + [v.value for v in self.all_constants[4:-2]]
        self.shared_property_23 = shared_property + [0] * (23 - len(shared_property) - 6) + [
            v.value for v in self.all_constants[-2:]] + [tw, step_max, neu_nums, rate]  # 6 = len([-2:]) + 4

        self.property = []
        for j in range(self.chip_num):
            prop = [j, self.ndma_staddr, self.npu_on_s[j]] + \
                self.shared_property_23
            self.property.append(prop)

        return self.smt_result, self.property

    def get_smt_96bit_result(self, tile_id, npu_id):
        self.config['tile_id'] = tile_id
        self.config['npu_id'] = npu_id
        self.smt_result = self.smt96_compiler.func_to_96bit_smt_cus()

        return self.smt_result

    def write_to_bin(self, save_dir):
        """Write hardware related to bin files
        """
        save_dir = Path(save_dir)

        # npu_ctrl
        output_dir = save_dir / 'npu_ctrl'
        output_dir.mkdir(exist_ok=True, parents=True)
        for i in range(self.chip_num):
            file_path = output_dir / f'npu_after_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path,
                        arr=self.npu_after_s[i], dtype="<u4")

        for i in range(self.chip_num):
            file_path = output_dir / f'npu_all0_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path,
                        arr=self.npu_all0_s[i], dtype="<u4")

        for i in range(self.chip_num):
            file_path = output_dir / f'npu_spike_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path,
                        arr=self.npu_spike_s[i], dtype="<u4")

        for i in range(self.chip_num):
            file_path = output_dir / f'npu_on_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path, arr=self.npu_on_s[i], dtype="<u4")

        # npu staddr
        output_dir = save_dir / 'ndma_staddr'
        output_dir.mkdir(exist_ok=True, parents=True)
        for i in range(self.chip_num):
            file_path = output_dir / f'ndma_staddr_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path, arr=self.ndma_staddr, dtype="<u4")

        # tile id
        output_dir = save_dir / 'tile_id'
        output_dir.mkdir(exist_ok=True, parents=True)
        for i in range(self.chip_num):
            file_path = output_dir / f'tile_id_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path, arr=i, dtype="<u4")

        # remote4
        output_dir = save_dir / 'remote4'
        output_dir.mkdir(exist_ok=True, parents=True)
        for i in range(self.chip_num):
            file_path = output_dir / f'remote4_{i}.bin'
            file_path.unlink(missing_ok=True)
            Fake.fwrite(file_path=file_path, arr=[0, 1], dtype="<u4")

        # spdma
        output_dir = save_dir / 'spdma'
        output_dir.mkdir(exist_ok=True, parents=True)
        for i in range(self.chip_num):
            file_path = output_dir / f'spdma_{i}.bin'
            file_path.unlink(missing_ok=True)
            # 655376--'000A_0010'  327688:'5_0008'  131088：‘2_0010’
            Fake.fwrite(file_path=file_path, arr=self.spdma_thres, dtype="<u4")

        # smt 32bit
        if not self.asic_flag:
            # 32bit hw result
            _ = self.get_smt_32bit_result()

            # property
            output_dir = save_dir / 'property'
            output_dir.mkdir(exist_ok=True, parents=True)
            for i in range(self.chip_num):
                file_path = output_dir / f'property_{i}.bin'
                file_path.unlink(missing_ok=True)
                Fake.fwrite(file_path=file_path,
                            arr=self.property[i], dtype="<u4")

            # smt 32bit
            output_dir = save_dir / 'smt_32bit'
            output_dir.mkdir(exist_ok=True, parents=True)
            instr_bin_all = np.zeros(1024, dtype=np.uint32)
            for n in range(1024):
                if n < len(self.smt_result):
                    instr_bin = ''.join(self.smt_result[n].value)
                    value = int(instr_bin, 2)
                    instr_bin_all[n] = value
            for i in range(self.chip_num):
                file_path = output_dir / f'smt_{i}.bin'
                file_path.unlink(missing_ok=True)
                Fake.fwrite(file_path=file_path,
                            arr=instr_bin_all, dtype="<u4")

        # smt 96bit
        if self.asic_flag:
            output_dir = save_dir / 'smt_96bit'
            output_dir.mkdir(exist_ok=True, parents=True)
            for tile_id in range(self.chip_num):
                # FIXME, not necessary to write hex files
                # hex_file_path = output_dir / f'smt.hex'
                bin_file_path = output_dir / f'smt_{tile_id}.bin'
                if bin_file_path.exists():
                    bin_file_path.unlink()

                for npu_id in range(min(self.npu_num, 16)):
                    smt_result = self.get_smt_96bit_result(tile_id, npu_id)
                    hex_data = []
                    for line in smt_result:
                        instr_bin = ''.join(
                            line.bin_value_for_human.split('_'))
                        parts = [instr_bin[i:i+32]
                                 for i in reversed(range(0, len(instr_bin), 32))]
                        # ['0000021F', '0077F400', '02400000']
                        hex_parts = [format(int(part, 2), '08X')
                                     for part in parts]
                        # ['00000000']
                        padding = ['\\n'.join(['0' * 8])]
                        hex_data.extend(hex_parts + padding)

                    # with open(bin_file_path, 'ab') as f_out, open(hex_file_path, 'wt') as f_tmp:
                    #     for item in hex_data:
                    #         data_val = int(item, 16)
                    #         data_arr = np.array([data_val])
                    #         data_arr.astype("<u4").T.tofile(f_out)
                    #         f_tmp.write(item + '\n')
