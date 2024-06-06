"""SMT 96位指令"""

from __future__ import annotations

import re
from typing import Union

from pyparsing import Suppress, Word, nested_expr, nums

from ...common.smt_96_base import REG_INDEX
from ...common.smt_96_reg import Register96, RegisterCollection
from ..smt_96_op import OPCode, OPField, OPType
from .smt96 import SMT96, SMT96ASMParseError


class SRAM_OP(SMT96):
    """SRAM 操作

    Example:
        >>> regs = RegisterCollection()
        >>> SRAM_OP("R1 = SRAM[3], R2 = SRAM[4], R3 = SRAM[5]", regs)
        R1 = SRAM[0], R2 = SRAM[1], R3 = SRAM[2]
        >>> SRAM_OP("SRAM[3]=R1, SRAM[4] = R2", regs)
        SRAM[0] = R1, SRAM[1] = R2
    """

    def __init__(self, expr: str, regs: RegisterCollection) -> None:
        """构造函数

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """
        super().__init__(op_type=OPType.SRAM_OP)

        if not expr:  # empty object
            return

        dst_src = self.parse_expr(expr, regs)[0]  # 只取第一个表达式
        self.op_0 = OPCode.SRAM_SAVE if isinstance(dst_src[0][0], int) else OPCode.SRAM_LOAD
        self.op_1 = self.op_0
        self.field_0 = OPField(op_code=self.op_0)
        self.field_1 = OPField(op_code=self.op_1)

        for i, d_s in enumerate(dst_src):
            dst, src = d_s
            v = dst if self.op_0 == OPCode.SRAM_LOAD else src
            self.fields[int(i / 4)].update_field(v, i % 4)

    @property
    def asm_value(self) -> str:
        """汇编语句

        Returns:
            str: 汇编语句
        """
        asm = []
        for field in self.fields:
            for i, reg in enumerate(field.raw_fields):
                if isinstance(reg, str):  # 默认值 "00000"
                    continue
                if reg.index == REG_INDEX.NONE_REG:
                    continue
                reg = SMT96.get_asm_value(reg)
                if self.op_0 == OPCode.SRAM_LOAD:
                    asm.append(f"{reg} = SRAM[{i}]")
                else:
                    asm.append(f"SRAM[{i}] = {reg}")

        return ", ".join(asm)

    @classmethod
    # pylint: disable-next=too-many-locals
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[list[Union[Register96, int]]]:
        """解析表达式

        Example:
            >>> regs = RegisterCollection()
            >>> SRAM_OP.parse_expr("R1 = SRAM[3], R2 = SRAM[4], R3 = SRAM[5]", regs)
            [[[<R1 = 0, used by: []>, 3], [<R2 = 0, used by: []>, 4], [<R3 = 0, used by: []>, 5]]]
            >>> SRAM_OP.parse_expr("SRAM[3] = R5, SRAM[4] = R5", regs)
            Traceback (most recent call last):
            ...
            snn_compiler.backend.smt_96_stmt.smt96.SMT96ASMParseError: 源寄存器重复 ...
            >>> SRAM_OP.parse_expr("R1 = SRAM[3], SRAM[4] = R1, R3 = SRAM[5]", regs)
            Traceback (most recent call last):
            ...
            snn_compiler.backend.smt_96_stmt.smt96.SMT96ASMParseError: SRAM 操作不一致, ...
            >>> SRAM_OP.parse_expr("R1 = SRAM[3]; SRAM[4] = R2; R3 = SRAM[5]", regs)
            [[[<R1 = 0, used by: []>, 3]], [[4, <R2 = 0, used by: []>]], [[<R3 = 0, used by: []>, 5]]]

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list[list[list[Union[Register96, int]]]]: 解析后的 SRAM 指令列表
        """
        register = SMT96.get_register_parser(regs)
        sram = Suppress("SRAM") + nested_expr("[", "]", content=Word(nums))
        sram.set_parse_action(lambda t: int(t[0][0]))

        load = register + Suppress("=") + sram
        save = sram + Suppress("=") + register
        sram_operation = load | save

        result = []
        for one_expr in re.split(r";|\n", expr):
            one_result = []
            expr_type = ""
            src_set, dst_set = set(), set()
            for one_op in one_expr.split(","):
                if not (one_op := one_op.strip()):
                    continue
                dst, src = sram_operation.parse_string(one_op, parse_all=True).as_list()

                # if dst in dst_set:
                #     raise SMT96ASMParseError(f"目的寄存器重复 {one_expr = }")

                # if src in src_set:
                #     raise SMT96ASMParseError(f"源寄存器重复 {one_expr = }")

                dst_set.add(dst)
                src_set.add(src)

                current_type = "load" if isinstance(dst, Register96) else "save"

                if not expr_type:
                    expr_type = current_type

                if expr_type != current_type:
                    raise SMT96ASMParseError(f"SRAM 操作不一致, 必须全为 SRAM 到寄存器或者寄存器到 SRAM. {one_expr = }")

                one_result.append([dst, src])

                if len(one_result) > 8:
                    raise SMT96ASMParseError(f"SRAM 操作支持最多 8 个寄存器, {one_expr = }")

            result.append(one_result)
        return result


__all__ = ["SRAM_OP"]
