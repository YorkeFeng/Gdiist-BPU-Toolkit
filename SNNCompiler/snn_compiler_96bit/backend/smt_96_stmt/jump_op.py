"""SMT 96位指令"""

from __future__ import annotations

import re

import pyparsing as pp
from pyparsing import Group, Optional, Suppress, Word, delimited_list, nums

from ...common.smt_96_base import CTRL_PULSE, IEEE754
from ...common.smt_96_reg import RegisterCollection
from ..smt_96_op import OPCode, OPField, OPType
from .smt96 import SMT96, SMT96ASMParseError


class JUMP_OP(SMT96):
    """跳转操作

    Example:
        >>> regs = RegisterCollection()
        >>> JUMP_OP("JUMP 1, 2, 3", regs).asm_value
        'JUMP(2) 1, 2, 3'
        >>> JUMP_OP("JUMP 1, 2, 3; JUMP 3, 2, 1", regs).asm_value
        'JUMP(2) 1, 2, 3'
        >>> JUMP_OP("JUMP 0, 0, 1", regs).asm_value
        'JUMP(2) 0, 0, 1'
        >>> JUMP_OP("JUMP 0, 0, -1", regs).asm_value
        'JUMP(2) 0, 0, -1'

    """

    def __init__(self, expr: str, regs: RegisterCollection) -> None:
        """构造函数

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """

        def to_field(o: int) -> str:
            if abs(o) >= (1 << 8):
                raise ValueError(f"不支持的 JUMP 指令 {expr = }, {abs(o) = } >= 256")
            field_result = "1" if o < 0 else "0"
            field_result += f"{abs(o):08b}"
            return field_result

        super().__init__(op_type=OPType.REG_OP)

        if not expr:  # empty object
            return

        expr = re.split(r";|\n", expr)[0]  # 只取第一个表达式
        v, a, b, c = self.parse_expr(expr, regs)[0]
        a, b, c = map(to_field, [a, b, c])
        imm = IEEE754(f"00000{a}{b}{c}")
        op_0 = op_1 = OPCode.IMM_SET
        field_0 = OPField(OPCode.IMM_SET, fields=[imm, regs.NONE_REG])
        field_1 = OPField(OPCode.IMM_SET, fields=[v, regs.CTRL_PULSE])
        self.op_0 = op_0
        self.field_0 = field_0
        self.op_1 = op_1
        self.field_1 = field_1

    @classmethod
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[list[int]]:
        """解析表达式, ; 或 \\n 分隔多个表达式

        Example:
            >>> regs = RegisterCollection()
            >>> JUMP_OP.parse_expr("JUMP 1, 2, -3", regs)
            [[2, 1, 2, -3]]
            >>> JUMP_OP.parse_expr("JUMP 1, 2, 3; JUMP 1, 3, 2", regs)
            [[2, 1, 2, 3], [2, 1, 3, 2]]
            >>> JUMP_OP.parse_expr("JUMP(130) 1, 2, -3", regs)
            [[130, 1, 2, -3]]

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list[list[int]]: 解析后的指令列表, e.g. [[1, 2, 3]]
        """
        offset = Group(Optional("-") + SMT96.get_constant_parser())
        value = Word(nums).set_parse_action(lambda t: int(t[0]))
        jump = Suppress("JUMP") + Group(Optional(Suppress("(") + value + Suppress(")"))) + delimited_list(offset, ",")

        result = []
        try:
            for one_expr in re.split(r";|\n", expr):
                v, a, b, c = jump.parse_string(one_expr, parse_all=True).as_list()
                if not v:
                    v = CTRL_PULSE.SMT_JUMP.value
                else:
                    v = v[0]
                a = int("".join(map(str, a)))
                b = int("".join(map(str, b)))
                c = int("".join(map(str, c)))
                if not all(isinstance(i, int) for i in [a, b, c]):
                    raise SMT96ASMParseError(f"不支持的 JUMP_OP 操作 {expr}")
                result.append([v, a, b, c])
            return result
        except pp.ParseException as exc:
            raise SMT96ASMParseError(f"不支持的 JUMP_OP 操作 {expr}") from exc

    @property
    def asm_value(self) -> str:
        """汇编语句

        Example:
            >>> regs = RegisterCollection()
            >>> JUMP_OP("JUMP 1, 2, 3", regs).asm_value
            'JUMP(2)  1, 2, 3'

        Returns:
            str: 汇编语句
        """

        def to_int(f: str) -> int:
            return -int(f[1:], 2) if f[0] == "1" else int(f[1:], 2)

        bin_value = self.field_0.raw_fields[0].bin_value
        a = bin_value[5 : 5 + 9]
        b = bin_value[5 + 9 : 5 + 9 + 9]
        c = bin_value[5 + 9 + 9 :]
        a, b, c = map(to_int, [a, b, c])
        v = self.field_1.raw_fields[0]
        return f"JUMP({v}) {a}, {b}, {c}"


__all__ = ["JUMP_OP"]
