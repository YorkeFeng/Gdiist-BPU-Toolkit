"""这个模块包含了用于处理 SMT 96 位指令的常用函数和类
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backend.smt_96_stmt.reg_op import REG_OP

from ..common.smt_96_reg import Register96, RegisterCollection, RegOrConstant
from .smt_96_stmt import FP_OP, SMT96


@dataclass
class SMT96Factory:
    """96-bit SMT 语句生成器."""

    regs: RegisterCollection = field(default_factory=RegisterCollection)
    """寄存器集合"""

    def get_expr(self, a: RegOrConstant, b: RegOrConstant, c: Register96, op: str) -> list[SMT96]:
        """获取表达式

        - 寄存器对象不变
        - 常数转换为 IEEE754 格式

        >>> regs = RegisterCollection()
        >>> SMT96Factory(regs).get_expr(0, regs[3], regs[0], "+")
        'R0 = 0 + R3'
        >>> SMT96Factory(regs).get_expr(0, regs[4], regs[1], "+")
        'R1 = 0 + R4'
        >>> SMT96Factory(regs).get_expr(regs[6], regs[3], regs[2], "+")
        'R2 = R6 + R3'

        Args:
            a RegOrConstant: 第一个操作数
            b RegOrConstant: 第二个操作数
            c (Register, optional): 结果寄存器
            op (str): 操作符

        Returns:
            str: 表达式
        """
        if not isinstance(c, Register96):
            raise ValueError(f"表达式不支持类型 {c = }, 只支持 Register96")

        a = SMT96.get_asm_value(a)
        b = SMT96.get_asm_value(b)
        return f"R{c.index} = {a} {op} {b}"

    def add(self, a: RegOrConstant, b: RegOrConstant, c: Register96) -> list[SMT96]:
        """加法 c = a + b. MUL_P = 0 * 0

        >>> regs = RegisterCollection()
        >>> SMT96Factory(regs).add(0, regs[3], regs[0])
        [R0 = 0.00 + R3]
        >>> SMT96Factory(regs).add(0, regs[4], regs[1])
        [R1 = 0.00 + R4]
        >>> SMT96Factory(regs).add(regs[6], regs[3], regs[2])
        [R2 = R6 + R3]

        Args:
            a RegOrConstant: 被加数寄存器或者数值
            b RegOrConstant: 加数寄存器
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """
        return [FP_OP(self.get_expr(a, b, c, "+"), self.regs)]

    def subtract(self, a: RegOrConstant, b: RegOrConstant, c: Register96) -> list[SMT96]:
        """减法 c = a - b. MUL_P = 0 * 0

        >>> regs = RegisterCollection()
        >>> SMT96Factory(regs).subtract(0, regs[3], regs[0])
        [R0 = 0.00 - R3]
        >>> SMT96Factory(regs).subtract(0, regs[4], regs[1])
        [R1 = 0.00 - R4]
        >>> SMT96Factory(regs).subtract(regs[6], regs[3], regs[2])
        [R2 = R6 - R3]

        Args:
            a RegOrConstant: 被减数寄存器或者数值
            b RegOrConstant: 减数寄存器
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT96]: SMT 语句
        """
        return [FP_OP(self.get_expr(a, b, c, "-"), self.regs)]

    def multiply(self, a: RegOrConstant, b: RegOrConstant, c: Register96) -> list[SMT96]:
        """乘法 c = a * b. MUL_P = 0 * 0

        >>> regs = RegisterCollection()
        >>> SMT96Factory(regs).multiply(0, regs[3], regs[0])
        [R0 = 0.00 * R3]
        >>> SMT96Factory(regs).multiply(0, regs[4], regs[1])
        [R1 = 0.00 * R4]
        >>> SMT96Factory(regs).multiply(regs[6], regs[3], regs[2])
        [R2 = R6 * R3]

        Args:
            a RegOrConstant: 被乘数寄存器或者数值
            b RegOrConstant: 乘数寄存器
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT96]: SMT 语句
        """
        return [FP_OP(self.get_expr(a, b, c, "*"), self.regs)]

    def move(self, src: Register96, dst: Register96) -> list[SMT96]:
        """移动寄存器 src -> dst

        >>> regs = RegisterCollection()
        >>> SMT96Factory(regs).move(regs[1], regs[0])
        [R1 -> R0]
        >>> SMT96Factory(regs).move(regs[3], regs[2])
        [R3 -> R2]

        Args:
            src (Register): 源寄存器
            dst (Register): 目标寄存器

        Returns:
            list[SMT96]: SMT 语句
        """
        return [REG_OP(f"R{dst.index} = R{src.index}", self.regs)]

__all__ = ["SMT96Factory"]
