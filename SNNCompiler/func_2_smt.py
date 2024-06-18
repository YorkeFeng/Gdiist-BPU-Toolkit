import math

import brainpy as bp
from brainpy._src.dyn.neurons.base import GradNeuDyn

from .snn_compiler_32bit.ir_to_smt import SMTCompiler
from .snn_compiler_32bit.smt import Number, Operator, Register
from .snn_compiler_96bit.backend.smt_96_stmt.smt96 import SMT96
from .snn_compiler_96bit.common.smt_96_base import (CTRL_LEVEL, CTRL_PULSE,
                                                    IBinary)
from .snn_compiler_96bit.flow.smt_96_compiler import SMT96Compiler


def get_v_func(model):
    for ds in model.nodes().subset(GradNeuDyn).values():
        return ds.integral.f


def func_to_32bit_smt(model, cv) -> None:
    """
    Args:
        model: E-I模型,
        cv: config parameters.
    """

    if bp.__version__ < '2.4.5':
        # FIXME, 确定是哪个brainpy版本, 根据版本号确定？还是考虑向下兼容？确认后集成到func_to_smt函数
        # NEW brainpy version: net = EINet(bp.DynSysGroup)
        index = 0
        for proj in model.nodes().subset(bp.Projection).values():
            if proj.refs["post"] != model.I:
                continue
            index += 1
            cv[f"E{index}"] = proj.refs["out"].E
            cv[f"tau{index}"] = proj.refs["syn"].tau

    elif bp.__version__ >= '2.4.5':
        # OLD brainpy version: net = EINet_v1(bp.Network)
        for proj in model.nodes().subset(bp.synapses.Exponential).values():
            if proj.pre == model.I:
                index = 1
                cv[f"E{index}"] = proj.output.E
                cv[f"tau{index}"] = proj.syn.tau
            else:
                index = 2
                cv[f"E{index}"] = proj.output.E
                cv[f"tau{index}"] = proj.syn.tau

    # only one
    v_func = get_v_func(model)

    i_func = {
        "I": lambda g1, g2, V: cv['R'] * g1 * (cv['E1'] - V) + cv['R'] * g2 * (cv['E2'] - V),
    }

    g_func = {
        "g1": lambda g1: g1 * math.exp(-1 / cv['tau1']),
        "g2": lambda g2: g2 * math.exp(-1 / cv['tau2']),
    }

    func = {"V": v_func}
    func.update(g_func)

    i_compiler, v_compiler, smt_result = SMTCompiler.compile_all(
        func=func,
        preload_constants=cv,
        i_func=i_func["I"],
        reg_map={"g1": "R3", "g2": "R4"},
        update_method={"g1": "update", "g2": "update", "V": "acc"}
    )

    # 预载入常数, write to property
    all_constants = i_compiler.preload_constants | v_compiler.preload_constants
    printed_name = []
    all_constants_new1 = []
    all_constants_new = sorted(all_constants, key=lambda r: int(r.name[2:]))
    for pc in all_constants_new:
        if pc.name not in printed_name:
            printed_name.append(pc.name)
            all_constants_new1.append(pc)

    return smt_result, all_constants_new1, cv


class SMT96_Aisc_Base:
    def __init__(self, model, cv):
        self.model = model
        self.cv = cv
        self.cv['total_step'] = 0
        self.cv['I'] = 2.0
        self.cv['weight_phase'] = 33554432

        self.smt_result, self.v_compiler = self.func_to_96bit_smt_init()

    def func_to_96bit_smt_cus(self,):

        pre_func = f"""
            03: R_CHIP_NPU_ID = {self.cv['npu_id'] }, R_NEU_NUMS = {self.cv['neuron_num']}
        """

        pre_func_stmts = list(SMT96.create_from_expr(
            pre_func, regs=self.v_compiler.regs))

        self.smt_result[2] = pre_func_stmts[0]

        return self.smt_result

    def func_to_96bit_smt_init(self,):
        """
    Args:
        model: E-I模型,
        cv: config parameters.
    """
        if bp.__version__ < '2.4.5':
            # NEW brainpy version: net = EINet(bp.DynSysGroup)
            index = 0
            for proj in self.model.nodes().subset(bp.Projection).values():
                if proj.refs["post"] != self.model.I:
                    continue
                index += 1
                self.cv[f"E{index}"] = proj.refs["out"].E
                self.cv[f"tau{index}"] = proj.refs["syn"].tau

            for ds in self.model.nodes().subset(GradNeuDyn).values():
                self.cv["R"] = ds.R
                self.cv["V_reset"] = ds.V_reset
                self.cv["V_rest"] = ds.V_rest
                self.cv["V_th"] = ds.V_th
                self.cv["tau_ref"] = ds.tau_ref

        elif bp.__version__ >= '2.4.5':
            # OLD brainpy version: net = EINet_v1(bp.Network)
            for proj in self.model.nodes().subset(bp.synapses.Exponential).values():
                if proj.pre == self.model.E:
                    index = 1
                    self.cv[f"E{index}"] = proj.output.E
                    self.cv[f"tau{index}"] = proj.syn.tau
                else:
                    index = 2
                    self.cv[f"E{index}"] = proj.output.E
                    self.cv[f"tau{index}"] = proj.syn.tau

            for ds in self.model.nodes().subset(GradNeuDyn).values():
                self.cv["R"] = ds.R
                self.cv["V_reset"] = ds.V_reset
                self.cv["V_rest"] = ds.V_rest
                self.cv["V_th"] = ds.V_th
                self.cv["tau_ref"] = ds.tau_ref

        # only one
        v_func = get_v_func(self.model)

        i_func = {
            "I": lambda g1, g2, V: self.cv['R'] * g1 * (self.cv['E1'] - V) + self.cv['R'] * g2 * (self.cv['E2'] - V),
        }

        g_func = {
            "g1": lambda g1: g1 * math.exp(-1 / self.cv['tau1']),
            "g2": lambda g2: g2 * math.exp(-1 / self.cv['tau2']),
        }

        # ASIC版本
        funcs = {"delta_V": v_func}
        funcs.update(g_func)
        funcs.update(i_func)

        # from MATLAB code
        cfg = 0
        rnd = 0
        rate = 0
        # scope = 256
        # total_num = 1024 * scope
        total_num = self.cv['neuron_num']
        if total_num > 1023:
            neu_num = 1024
        else:
            neu_num = total_num

        neu_num = IBinary.dec2bin(neu_num, 11)
        cfg_bin = IBinary.dec2bin(cfg, 13 - 11)
        rnd_bin = IBinary.dec2bin(rnd, 17 - 13)
        rate_bin = IBinary.dec2bin(rate, 32 - 17)
        neu_num_bin = rate_bin + rnd_bin + cfg_bin + neu_num

        neu_num = IBinary(0, 32)
        neu_num.bin_value = neu_num_bin
        neu_num = neu_num.dec_value
        total_step = self.cv['total_step']
        # vth = -50.0
        # trst = 6
        # vrst = -60.0
        vth = self.cv["V_th"]
        # trst = self.cv["tau_ref"]
        trst = int(self.cv["tau_ref"]) + 1
        vrst = self.cv["V_reset"]
        Ix = self.cv["I"]

        self.cv['neuron_num'] = neu_num

        ndma_phase = 32800
        weight_phase = self.cv['weight_phase']  # math.pow(2, 25), must be int

        constants = {"V_th": vth,
                     "I_x": Ix}  # 编译器输入的常数

        predefined_regs = {
            "zero0": "R0",
            "neu_en": "R1",
            "tlastsp": "R2",
            "V": "R3",
            "g1": "R4",
            "g2": "R5",
            # R6 is not used
            "w_en": "R8",
            "neu_id": "R9",
            "w1": "R10",
            "w2": "R11",
        }

        _, v_compiler, smt_result = SMT96Compiler.compile_all(
            funcs=funcs,
            constants=constants,
            predefined_regs=predefined_regs,
        )

        pre_func = f"""
            01: R_CTRL_LEVEL = CFG_EN, R_ZERO_REG = 0
            02: R_STEP_REG = {total_step}, R_PHASE = 0
            03: R_CHIP_NPU_ID = 0, R_NEU_NUMS = {neu_num}
            04: R_TRST_REG1 = {trst}, R_TRST_REG0 = {trst}  //uger debug #2.0
            05: R_VRST_REG1 = {vrst}, R_VRST_REG0 = {vrst}
            06: R_V_DIFF_REG1 = 0, R_V_DIFF_REG0 = 0
            07: R_TLASTSP_TMP1 = 0, R_TLASTSP_TMP0 = 0
            08: R_NONE_REG = 0, R_CTRL_PULSE = LFSR_INIT_SET //uger debug #2.1
            09: R_NONE_REG = 1, R_CTRL_PULSE = LFSR_SET //uger debug #2.2
            10: NOP
            11: NOP
            12: R_CTRL_LEVEL = NPU_RST_EN, R_NONE_REG = 0   //uger debug #2.3
            13: JUMP 0, 0, -1  // ndma rd wait
            14: NOP
            15: R_PHASE = {ndma_phase}, R_CTRL_PULSE = TIMER_SET
            16: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.NDMA_RD_EN}, R_NONE_REG = 0 // sim_en on & ndma_en on // uger debug #2.4
            17: JUMP 0, 0, -1  // ndma rd wait
            18: NOP
            19: R_PHASE = {weight_phase}, R_CTRL_PULSE = TIMER_SET  // set wacc
            20: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.W_EN}, R_NONE_REG = 0  // sim_en on & w_en on
            21: JUMP 13, -1, 1  // uger debug #1
            22: NOP
            23: R_w1 <= R_w_en, R_w2 <= R_neu_id, R_CTRL_PULSE = WRIGHT_RX_READY // uger debug #2
            24: NOP
            25: R_zero0 = SRAM[0], R_neu_en = SRAM[1], R_tlastsp = SRAM[2], R_V = SRAM[3], R_g1 = SRAM[4], R_g2 = SRAM[5], R_ZERO_REG = SRAM[6], R_ZERO_REG = SRAM[7]
            26: R_g1 = R_w1 + R_g1
            27: R_g2 = R_w2 + R_g2
            28: NOP
            29: NOP
            30: NOP
            31: NOP
            32: SRAM[0] = R_zero0, SRAM[1] = R_neu_en, SRAM[2] = R_tlastsp, SRAM[3] = R_V, SRAM[4] = R_g1, SRAM[5] = R_g2, SRAM[6] = R_ZERO_REG, SRAM[7] = R_ZERO_REG
            33: JUMP({CTRL_PULSE.SMT_JUMP + CTRL_PULSE.W_JUMP}) 0, 0, -13
            34: NOP
            35: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.V_EN}, R_CTRL_PULSE = NPU_SET
            36: JUMP 0, {2 + len(smt_result) + 3}, 1   //uger debug #3. FIX: jump to post_func: 86
            37: NOP
            38: R_zero0 = SRAM[0], R_neu_en = SRAM[1], R_TLASTSP_TMP0 = SRAM[2], R_V = SRAM[3], R_g1 = SRAM[4], R_g2 = SRAM[5], R_ZERO_REG = SRAM[6], R_ZERO_REG = SRAM[7]
        """

        print(pre_func)

        pre_func_stmts = list(SMT96.create_from_expr(
            pre_func, regs=v_compiler.regs))

        post_func = f"""
            83: R_tlastsp = R_TLASTSP_TMP0, R_NONE_REG = 0  //uger debug #2.6
            84: SRAM[0] = R_zero0, SRAM[1] = R_neu_en, SRAM[2] = R_tlastsp, SRAM[3] = R_V, SRAM[4] = R_g1, SRAM[5] = R_g2, SRAM[6] = R_ZERO_REG, SRAM[7] = R_ZERO_REG
            85: JUMP({CTRL_PULSE.SMT_JUMP + CTRL_PULSE.NPU_PLUS}) 0, 0, {-(len(pre_func_stmts) + len(smt_result) + 2 + 1 - 35)}
            86: NOP
            87: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.S_EN}, R_CTRL_PULSE = {CTRL_PULSE.NPU_SET}
            88: JUMP({CTRL_PULSE.SMT_JUMP + CTRL_PULSE.STEP_PLUS}) 0, {-(len(pre_func_stmts) + len(smt_result) + 5 + 1 - 18)}, 1
            89: NOP
            90: R_PHASE = {ndma_phase}, R_CTRL_PULSE = {CTRL_PULSE.TIMER_SET + CTRL_PULSE.NPU_SET}
            91: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN + CTRL_LEVEL.NDMA_WR_EN}
            92: JUMP 0, 0, -1  //uger debug #4
            93: NOP
            94: R_CTRL_LEVEL = {CTRL_LEVEL.SIM_EN}
            95: JUMP 0, 0, {-(len(pre_func_stmts) + len(smt_result) + 12 + 1 - 11)}
            96: R_CTRL_LEVEL = {CTRL_LEVEL.NPU_RST_EN}, R_CTRL_PULSE = {CTRL_PULSE.SIM_END} //uger debug #5
        """

        post_func_stmts = list(SMT96.create_from_expr(
            post_func, regs=v_compiler.regs))

        nops = "\n"
        for i in range(256 - (len(pre_func_stmts) + len(smt_result) + len(post_func_stmts))):
            nops += f"{i}: NOP\n"

        nops = list(SMT96.create_from_expr(nops, regs=v_compiler.regs))

        self.smt_result = pre_func_stmts + smt_result + post_func_stmts + nops

        self.v_compiler = v_compiler

        return self.smt_result, self.v_compiler
