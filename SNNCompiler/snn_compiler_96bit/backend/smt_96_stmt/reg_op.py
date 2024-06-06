"""SMT 96位指令"""

from __future__ import annotations

import math
import re
from typing import Union

import pyparsing as pp
from pyparsing import Group, Literal, Optional, Suppress, delimited_list, nested_expr

from ...common.smt_96_base import CTRL_PULSE, IEEE754, REG_INDEX, SMT96_FIELD_EMPTY
from ...common.smt_96_reg import Register96, RegisterCollection
from ..smt_96_op import OPCode, OPField, OPType
from .smt96 import SMT96, SMT96ASMParseError


class REG_OP(SMT96):
    """寄存器操作

    - REG_SET: 0: Register, 操作寄存器
    - IMM_SET: 1: Immediate, 操作数值
        - 转化成 IEEE754 对象
        - 使用 `IEEE754.raw_value` 属性保留原始值
    - BUS_SET: 2: Bus, 操作 Bus

    Example:
        >>> regs = RegisterCollection()
        >>> REG_OP("R2 = 332.0", regs)
        R2 = 332.00
        >>> REG_OP("R_CTRL_LEVEL = SMT_JUMP", regs)
        R_CTRL_LEVEL = CTRL_PULSE.SMT_JUMP
        >>> REG_OP("R2 = R3", regs)
        R2 = R3
        >>> REG_OP("R2 = 3", regs)
        R2 = 3
        >>> REG_OP("R2 = R3, R1 = 4", regs)
        R1 = 4, R2 = R3
    """

    def __init__(self, expr: str, regs: RegisterCollection) -> None:
        """构造函数

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """
        super().__init__(op_type=OPType.REG_OP)

        if not expr:  # empty object
            return

        self.field_0 = OPField(op_code=OPCode.REG_SET).empty_field
        self.field_1 = OPField(op_code=OPCode.REG_SET).empty_field

        expr = re.split(r";|\n", expr)[0]  # 只取第一条指令

        imm_set = []
        reg_set = []
        bus_set = []

        for dst, eq, src in self.parse_expr(expr, regs)[0]:
            if isinstance(src, IEEE754):
                imm_set.append((src, dst))
                continue
            if isinstance(src, Register96) and eq == "=":
                if not reg_set or len(reg_set[-1]) == 2:
                    reg_set.append([])
                reg_set[-1].append((src, dst))
                continue
            if isinstance(src, Register96) and eq == "<=":
                if not bus_set or len(bus_set[-1]) == 2:
                    bus_set.append([])
                bus_set[-1].append((src, dst))
                continue

            raise ValueError(f"不支持的类型 {dst} {eq} {src}")

        i = 0
        for k, v in {"bus_set": bus_set,  "reg_set": reg_set, "imm_set": imm_set}.items():   # uger debug #2024.04.30
            if not v:
                continue

            if k == "imm_set":
                for src, dst in v:
                    self.fields[i].op_code = OPCode.IMM_SET
                    self.fields[i].fields = [src, dst]
                    i += 1
                continue

            for src_dst_list in v:
                self.fields[i].op_code = OPCode.REG_SET if k == "reg_set" else OPCode.BUS_SET
                if len(src_dst_list) == 1:
                    src, dst = src_dst_list[0]
                    self.fields[i].fields = [src, regs.ZERO_REG, dst, regs.NONE_REG]
                else:  # len(src_dst_list) == 2
                    s0, d0 = src_dst_list[0]
                    s1, d1 = src_dst_list[1]
                    self.fields[i].fields = [s0, s1, d0, d1]
                i += 1

        self.op_0 = self.field_0.op_code
        self.op_1 = self.field_1.op_code

    @classmethod
    # pylint: disable-next=too-many-locals
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[list[Union[Register96, IEEE754]]]:
        """解析表达式. 如果常数带有小数点, 则转换成 IEEE 754 格式.

        Example:
            >>> regs = RegisterCollection()
            >>> REG_OP.parse_expr("R_CTRL_LEVEL = SMT_JUMP, R2 = R3, R1=3", regs)
            Traceback (most recent call last):
            ...
            snn_compiler.backend.smt_96_stmt.smt96.SMT96ASMParseError: 操作太多 ...
            >>> REG_OP.parse_expr("R_CTRL_LEVEL = SMT_JUMP, R2 = R3, R1 = R4", regs)
            [[[<CTRL_LEVEL = 0, used by: [-2]>, '=', <IEEE754:2:0.00>], [<R2 = 0, used by: []>, '=', <R3 = 0, used by: []>], [<R1 = 0, used by: []>, '=', <R4 = 0, used by: []>]]]
            >>> REG_OP.parse_expr("R2 <= R3", regs)
            [[[<R2 = 0, used by: []>, '<=', <R3 = 0, used by: []>]]]
            >>> REG_OP.parse_expr("R2 = 332", regs)
            [[[<R2 = 0, used by: []>, '=', <IEEE754:332:0.00>]]]
            >>> REG_OP.parse_expr("R2 = 4", regs)
            [[[<R2 = 0, used by: []>, '=', <IEEE754:4:0.00>]]]

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list[list[Union[Register96, IEEE754]]]: 解析后的指令列表, e.g. [R2, R3], [R2, 4]
        """

        constant = SMT96.get_constant_parser()
        register = SMT96.get_register_parser(regs)
        sign = Literal("+") | Literal("-")
        reg_or_constant = register | constant

        dst = register
        src = (
            Group(Optional(sign) + reg_or_constant)  # 可能是负数
            | Group(reg_or_constant)
            | nested_expr("(", ")", content=Optional(sign) + reg_or_constant)
        )
        eq = Literal("=") | Literal("<=")
        assignment = delimited_list(Group(dst + eq + src), delim=",")

        result = []
        try:
            # pylint: disable-next=duplicate-code
            for one_expr in re.split(r";|\n", expr):
                if not (one_expr := one_expr.strip()):
                    continue
                one_result = []
                count = {
                    "imm_set": 0,
                    "reg_set": 0,
                    "bus_set": 0,
                }
                for dst, eq, src in assignment.parse_string(one_expr, parse_all=True).as_list():
                    if isinstance(src[-1], Register96):  # 寄存器对象检查是否有负号
                        if src[0] == "-":
                            raise SMT96ASMParseError(f"不支持负数寄存器 {one_expr}")
                        if eq == "=":
                            count["reg_set"] += 0.5
                        else:
                            count["bus_set"] += 0.5
                    elif isinstance(src[-1], int):  # 整数常数转换成 IEEE 754
                        if eq != "=":
                            raise SMT96ASMParseError(f"不支持的操作 {one_expr}")
                        src[-1] = IEEE754(src[-1])
                        if src[0] == "-":
                            src[-1] = -src[-1]
                        count["imm_set"] += 1
                    elif isinstance(src[-1], IEEE754):
                        if eq != "=":
                            raise SMT96ASMParseError(f"不支持的操作 {one_expr}")
                        if src[0] == "-":
                            src[-1] = -src[-1]
                        count["imm_set"] += 1
                    else:
                        raise SMT96ASMParseError(f"不支持的类型 {src[-1] = } {type(src[-1]) = }")
                    src = src[-1]
                    one_result.append([dst, eq, src])

                    if sum(math.ceil(c) for c in count.values()) > 2:
                        raise SMT96ASMParseError(f"操作太多 {one_expr}")

                result.append(one_result)
            return result
        except pp.ParseException as exc:
            raise SMT96ASMParseError(f"不支持的 REG_OP 操作 {one_expr}") from exc

    @property
    def asm_value(self) -> str:
        """汇编语句

        Example:
            >>> regs = RegisterCollection()
            >>> REG_OP("R2 = 332.0", regs).asm_value
            'R2 = 332.00'
            >>> REG_OP("R2 <= 332.0", regs).asm_value
            Traceback (most recent call last):
            ...
            snn_compiler.backend.smt_96_stmt.smt96.SMT96ASMParseError: 不支持的操作 R2 <= 332.0
            >>> REG_OP("R2 <= R3", regs).asm_value
            'R2 <= R3'

        Returns:
            str: 汇编语句
        """
        asm = []
        eq = {
            OPCode.REG_SET: "=",
            OPCode.BUS_SET: "<=",
        }
        for one_field in self.fields:
            if one_field.is_empty:
                continue
            empty_field = SMT96_FIELD_EMPTY[one_field.op_code.name]

            try:
                if one_field.op_code in [OPCode.REG_SET, OPCode.BUS_SET]:
                    s0, s1, d0, d1 = one_field.dec_fields
                    if (s0 == d0) and (s1 == d1):
                        continue
                    if [s0, d0] != [empty_field[0], empty_field[2]]:
                        asm.append(f"R{d0} {eq[one_field.op_code]} R{s0}")
                    if [s1, d1] != [empty_field[1], empty_field[3]]:
                        asm.append(f"R{d1} {eq[one_field.op_code]} R{s1}")
                elif one_field.op_code == OPCode.IMM_SET:
                    src, dst = map(SMT96.get_asm_value, one_field.raw_fields)
                    asm.append(f"{dst} = {src}")
                else:
                    raise NotImplementedError(f"暂不支持 {one_field.op_code}")
            except ValueError as exc:
                raise ValueError(f"{one_field.op_code} {one_field.args} 不匹配") from exc

        if not asm:
            raise ValueError("没有 REG_OP 操作")

        return ", ".join(asm)


__all__ = ["REG_OP"]
