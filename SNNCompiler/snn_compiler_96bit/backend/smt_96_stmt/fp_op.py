# pylint: disable=line-too-long
"""FP_OP 指令

Example:
    >>> regs = RegisterCollection()
    >>> IEEE754(-4.0)
    <IEEE754:3229614080:-4.00>
    >>> FP_OP("R2 = -4 + R3", regs)
    R2 = -4.00 + R3
    >>> FP_OP("R2 = -4.0 + R3", regs)
    R2 = -4.00 + R3
    >>> regs = RegisterCollection()
    >>> program = '''
    ... R2 = R3 - 1.0
    ... R4 = R10 * R11
    ... '''
    >>> list(FP_OP.create_from_expr(program, regs))
    [R2 = -1.00 + R3, R4 = R10 * R11]
"""

from __future__ import annotations

import re
from typing import Union

import pyparsing as pp
from pyparsing import Group, Literal, Optional, Suppress, nested_expr

from ...common.smt_96_base import IEEE754
from ...common.smt_96_reg import Register96, RegisterCollection
from ..smt_96_op import OPCode, OPField, OPType
from .smt96 import SMT96, SMT96ASMParseError

EXP_OP_CODES: dict[str, tuple[str, Union[OPCode, None]]] = {
    "+i++r": ("", OPCode.IMM_ADD_POS),  # 数值加寄存器
    "+i+-r": ("", OPCode.IMM_ADD_NEG),  # 数值减寄存器
    "+i-+r": ("", OPCode.IMM_ADD_NEG),  # 数值减寄存器
    "+i--r": ("", OPCode.IMM_ADD_POS),  # 数值加寄存器
    "+i*+r": ("", OPCode.IMM_MUL_POS),  # 数值乘寄存器
    "+i*-r": ("", OPCode.IMM_MUL_NEG),  # 数值乘负寄存器
    "-i++r": ("", OPCode.IMM_ADD_POS),  # 数值加寄存器
    "-i+-r": ("", OPCode.IMM_ADD_NEG),  # 数值减寄存器
    "-i-+r": ("", OPCode.IMM_ADD_NEG),  # 数值减寄存器
    "-i--r": ("", OPCode.IMM_ADD_POS),  # 数值加寄存器
    "-i*+r": ("", OPCode.IMM_MUL_POS),  # 数值乘寄存器
    "-i*-r": ("", OPCode.IMM_MUL_NEG),  # 数值乘负寄存器
    "+r++r": ("", OPCode.REG_ADD_POS),  # 寄存器加寄存器
    "+r+-r": ("", OPCode.REG_ADD_NEG),  # 寄存器减寄存器
    "+r-+r": ("", OPCode.REG_ADD_NEG),  # 寄存器减寄存器
    "+r--r": ("", OPCode.REG_ADD_POS),  # 寄存器加寄存器
    "+r*+r": ("", OPCode.REG_MUL_POS),  # 寄存器乘寄存器
    "+r*-r": ("", OPCode.REG_MUL_NEG),  # 寄存器乘负寄存器
    "-r++r": ("swap", OPCode.REG_ADD_NEG),  # a b = b a 之后寄存器减寄存器
    "-r+-r": ("todo", None),
    "-r-+r": ("todo", None),
    "-r--r": ("swap", OPCode.REG_ADD_NEG),  # a b = b a 之后寄存器减寄存器
    "-r*+r": ("swap", OPCode.REG_MUL_NEG),  # a b = b a 之后寄存器乘负寄存器
    "-r*-r": ("", OPCode.REG_MUL_POS),  # 寄存器乘寄存器
    "+r++i": ("swap", OPCode.IMM_ADD_POS),  # a b = b a 之后数值加寄存器
    "-r++i": ("swap", OPCode.IMM_ADD_NEG),  # a b = b a 之后数值减寄存器
    "+r-+i": ("swap-", OPCode.IMM_ADD_POS),  # a b = -b a 之后数值加寄存器
    "-r-+i": ("swap-", OPCode.IMM_ADD_NEG),  # a b = -b a 之后数值减寄存器
    "+r*+i": ("swap", OPCode.IMM_MUL_POS),  # a b = b a 之后数值乘寄存器
    "-r*+i": ("swap", OPCode.IMM_MUL_NEG),  # a b = b a 之后数值乘负寄存器
    "+r+-i": ("swap", OPCode.IMM_ADD_POS),  # a b = b a 之后数值加寄存器
    "-r+-i": ("swap", OPCode.IMM_ADD_NEG),  # a b = b a 之后数值减寄存器
    "+r--i": ("swap+", OPCode.IMM_ADD_POS),  # a b = abs(b) a 之后数值加寄存器
    "-r--i": ("swap+", OPCode.IMM_ADD_NEG),  # a b = abs(b) a 之后数值加寄存器
    "+r*-i": ("swap", OPCode.IMM_MUL_POS),  # a b = b a 之后数值乘寄存器
    "-r*-i": ("swap", OPCode.IMM_MUL_NEG),  # a b = b a 之后数值乘负寄存器
}
"""表达式操作代码

- todo: 没有实现
- swap: a b = b a 之后实现
- swap-: a b = -b a 之后实现, 用于实现寄存器减数值
- swap+: a b = abs(b) a 之后实现, 用于实现寄存器减负数
- 其他: 直接实现
"""


class FP_OP(SMT96):
    """运算指令

    - IMM_ADD_POS: 0: Immediate, 加法, 正数
    - IMM_ADD_NEG: 1: Immediate, 加法, 负数
    - REG_ADD_POS: 2: Register, 加法, 正数
    - REG_ADD_NEG: 3: Register, 加法, 负数
    - IMM_MUL_POS: 4: Immediate, 乘法, 正数
    - IMM_MUL_NEG: 5: Immediate, 乘法, 负数
    - REG_MUL_POS: 6: Register, 乘法, 正数
    - REG_MUL_NEG: 7: Register, 乘法, 负数

    Example:
        >>> regs = RegisterCollection()
        >>> IEEE754(-4.0)
        <IEEE754:3229614080:-4.00>
        >>> FP_OP("R2 = -4 + R3", regs)
        R2 = -4.00 + R3
        >>> FP_OP("R2 = -4.0 + R3", regs)
        R2 = -4.00 + R3
        >>> regs = RegisterCollection()
        >>> program = '''
        ... R2 = R3 - 1.0
        ... R4 = R10 * R11
        ... '''
        >>> list(FP_OP.create_from_expr(program, regs))
        [R2 = -1.00 + R3, R4 = R10 * R11]
    """

    def __init__(self, expr: str, regs: RegisterCollection) -> None:
        """构造函数, 从表达式加载 `FP_OP` 对象, 只取第一个表达式

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """
        super().__init__(op_type=OPType.FP_OP)

        if not expr:  # empty object
            return

        self.field_0 = OPField(op_code=OPCode.IMM_ADD_POS).empty_field
        self.field_1 = OPField(op_code=OPCode.IMM_MUL_POS).empty_field

        add_part = mul_part = None

        expr = re.split(r";|\n", expr)[0]  # 只取第一个表达式
        dst, a, op, b = self.parse_expr(expr, regs)[0]

        if isinstance(a[-1], IEEE754) and isinstance(b[-1], IEEE754):
            raise NotImplementedError(f"暂不支持纯常数运算 {expr =}")

        a, op_code, b = self.preprocess_expr(a, op, b, expr)

        if op in "+-":
            add_part = (op_code, a, b, dst)
        elif op == "*":
            mul_part = (op_code, a, b, dst)
        else:
            raise SMT96ASMParseError(f"不支持的 FP_OP 操作 {expr = }")

        if add_part:
            self.field_0 = OPField(op_code=add_part[0], fields=add_part[1:])
        elif mul_part:
            self.field_1 = OPField(op_code=mul_part[0], fields=mul_part[1:])
        else:
            raise SMT96ASMParseError(f"没有 FP_OP 操作 {expr = }")

        self.op_0 = self.field_0.op_code
        self.op_1 = self.field_1.op_code

    @classmethod
    def preprocess_expr(
        cls,
        a: list[Union[Register96, int, IEEE754]],
        op: str,
        b: list[Union[Register96, int, IEEE754]],
        expr: str,
    ) -> tuple[Union[int, IEEE754], OPCode, int]:
        """预处理表达式

        - 变换操作数顺序
        - 常数转换为 `IEEE754` 对象
        - 寄存器编号提取

        Example:
            >>> FP_OP.preprocess_expr([3], "+", [Register96(4)], "R2 = 3 + R4")
            (<IEEE754:3:0.00>, <OPCode.IMM_ADD_POS: 'IMM_ADD_POS'>, <R4 = 0, used by: []>)
            >>> FP_OP.preprocess_expr([IEEE754(3.0)], "+", [Register96(4)], "R2 = 3.0 + R4")
            (<IEEE754:1077936128:3.00>, <OPCode.IMM_ADD_POS: 'IMM_ADD_POS'>, <R4 = 0, used by: []>)

        Args:
            a (list[str, Union[Register96, int, IEEE754]]): 操作数 a
            op (str): 操作符
            b (list[str, Union[Register96, int, IEEE754]]): 操作数 b
            expr (str): 表达式, 用于报错

        Returns:
            tuple[Union[int, IEEE754], OPCode, int]: 常数或寄存器编号, 操作符, 寄存器编号
        """
        fp_type = ""
        fp_type += "-" if a[0] == "-" else "+"
        fp_type += "r" if isinstance(a[-1], Register96) else "i"
        fp_type += op
        fp_type += "-" if b[0] == "-" else "+"
        fp_type += "r" if isinstance(b[-1], Register96) else "i"

        if fp_type not in EXP_OP_CODES:
            raise SMT96ASMParseError(f"不支持的 FP_OP 操作 {expr = }")

        action, op_code = EXP_OP_CODES[fp_type]

        if action == "todo":
            raise NotImplementedError(f"暂不支持寄存器运算 {expr =}")

        if action == "swap":  # 交换 a, b 顺序实现相同结果
            a, b = b, a
        elif action == "swap-":  # 实现寄存器加负数, b[-1] 是负数的绝对值
            a, b = [-(b[-1])], a
        elif action == "swap+":  # 实现寄存器减负数, b[-1] 是负数的绝对值
            a, b = [b[-1]], a
        elif action == "":
            pass
        else:
            raise NotImplementedError(f"不支持的 FP_OP 预处理操作 {action = }")
        # endregion: FP_OP 预处理

        if isinstance(a[-1], (int, IEEE754)):  # 常数数值统一转换成 IEEE754 构造函数中处理
            a = IEEE754(a[-1] if a[0] != "-" else -a[-1])
        elif isinstance(a[-1], Register96):  # 寄存器编号
            a = a[-1]
        else:
            raise SMT96ASMParseError(f"不支持的 FP_OP 操作 {expr = }, {a = }")

        # 经过预处理之后 b 只可能是寄存器数组
        b = b[-1]

        return a, op_code, b

    @classmethod
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[list[Union[str, int]]]:
        """解析表达式为 dst, a, op, b 型式.

        - 浮点数常数转换成 `IEEE754` 对象
        - 整数常数被看作为浮点数转换成 `IEEE754` 对象
        - 常数和寄存器的符号提取出来, 之后变换表达式的时候需要用到
        - R* 直接转换成寄存器对象

        Example:
            >>> regs = RegisterCollection()
            >>> FP_OP.parse_expr("R2 = -3 + (-R3)", regs)
            [[<R2 = 0, used by: []>, ['-', <IEEE754:1077936128:3.00>], '+', ['-', <R3 = 0, used by: []>]]]
            >>> FP_OP.parse_expr("R2 = -3 + R3", regs)
            [[<R2 = 0, used by: []>, ['-', <IEEE754:1077936128:3.00>], '+', [<R3 = 0, used by: []>]]]
            >>> FP_OP.parse_expr("R2 = -3 + R_NEU_NUMS", regs)
            [[<R2 = 0, used by: []>, ['-', <IEEE754:1077936128:3.00>], '+', [<NEU_NUMS = 0, used by: [-2]>]]]
            >>> FP_OP.parse_expr("R2 = -3 + R3; R4 = 0 + R3", regs)
            [[<R2 = 0, used by: []>, ['-', <IEEE754:1077936128:3.00>], '+', [<R3 = 0, used by: []>]], [<R4 = 0, used by: []>, [<IEEE754:0:0.00>], '+', [<R3 = 0, used by: []>]]]

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list[list[Union[str, int]]]: 解析后的指令列表, `(src, dst)` 列表
        """

        constant = SMT96.get_constant_parser()
        register = SMT96.get_register_parser(regs)
        reg_or_constant = register | constant
        sign = Literal("+") | Literal("-")

        dst = register
        a = (
            Group(Optional(sign) + reg_or_constant)  # 符号提取出来, 之后运算需要
            | Group(reg_or_constant)
            | nested_expr("(", ")", content=Optional(sign) + reg_or_constant)
        )
        b = a
        op = Literal("+") | Literal("-") | Literal("*")
        assignment = dst + Suppress("=") + a + op + b

        result = []
        try:
            for one_expr in re.split(r";|\n", expr):
                if not (one_expr := one_expr.strip()):
                    continue
                dst, a, op, b = assignment.parse_string(one_expr, parse_all=True).as_list()
                # 数值统一转换成 IEEE754 对象, 因为表达式中的常数不会是寄存器编号
                if isinstance(a[-1], (float, int)):
                    a[-1] = IEEE754(float(a[-1]))
                if isinstance(b[-1], (float, int)):
                    b[-1] = IEEE754(float(a[-1]))
                result.append([dst, a, op, b])
            return result
        except pp.ParseException as exc:
            raise SMT96ASMParseError(f"不支持的 FP_OP 操作 {one_expr}") from exc

    @property
    def asm_value(self) -> str:
        """汇编语句

        Example:
            >>> regs = RegisterCollection()
            >>> program = '''
            ... R3 = 2 + R4
            ... R3 = -2 - R4
            ... R3 = R2 + R4
            ... R3 = R2 - R4
            ... R3 = 2 * R4
            ... R3 = -2 * R4
            ... R3 = R2 * R4
            ... R3 = -R2 * R4
            ... R3 = R4 * (-R2)
            ... '''
            >>> for op in FP_OP.create_from_expr(program, regs):
            ...     op.asm_value
            'R3 = 2.00 + R4'
            'R3 = -2.00 - R4'
            'R3 = R2 + R4'
            'R3 = R2 - R4'
            'R3 = 2.00 * R4'
            'R3 = -2.00 * R4'
            'R3 = R2 * R4'
            'R3 = R4 * (-R2)'
            'R3 = R4 * (-R2)'
            >>> program_code = "; ".join(op.asm_value for op in FP_OP.create_from_expr(program, regs))
            >>> program_code_code = "; ".join(op.asm_value for op in FP_OP.create_from_expr(program_code, regs))
            >>> program_code_code == program_code
            True

        Returns:
            str: 汇编语句
        """
        asm = []

        if not self.field_0.is_empty:
            a, b, c = self.field_0.raw_fields
            a = SMT96.get_asm_value(a)
            b = SMT96.get_asm_value(b)
            c = SMT96.get_asm_value(c)
            asm.append(
                {
                    OPCode.IMM_ADD_POS: f"{c} = {a} + {b}",
                    OPCode.IMM_ADD_NEG: f"{c} = {a} - {b}",
                    OPCode.REG_ADD_POS: f"{c} = {a} + {b}",
                    OPCode.REG_ADD_NEG: f"{c} = {a} - {b}",
                }[self.field_0.op_code]
            )

        if not self.field_1.is_empty:
            a, b, c = self.field_1.raw_fields
            a = SMT96.get_asm_value(a)
            b = SMT96.get_asm_value(b)
            c = SMT96.get_asm_value(c)
            asm.append(
                {
                    OPCode.IMM_MUL_POS: f"{c} = {a} * {b}",
                    OPCode.IMM_MUL_NEG: f"{c} = {a} * (-{b})",
                    OPCode.REG_MUL_POS: f"{c} = {a} * {b}",
                    OPCode.REG_MUL_NEG: f"{c} = {a} * (-{b})",
                }[self.field_1.op_code]
            )

        if not asm:
            raise ValueError("没有 FP_OP 操作")

        return ", ".join(asm)


__all__ = ["FP_OP"]
