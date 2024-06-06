"""SMT 96位指令"""

from __future__ import annotations

from ...common.smt_96_base import SMT96_FIELD_EMPTY

from ...common.smt_96_reg import RegisterCollection
from ..smt_96_op import OPCode, OPField, OPType
from .smt96 import SMT96, SMT96ASMParseError


class NOP(SMT96):
    """NOP 指令

    Example:
        >>> regs = RegisterCollection()
        >>> NOP()
        NOP
        >>> list(NOP.create_from_expr("NOP; NOP; NOP", regs))
        [NOP, NOP, NOP]
    """

    # pylint: disable-next=unused-argument
    def __init__(self, expr: str = None, regs: RegisterCollection = None) -> None:
        """构造函数

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合
        """
        super().__init__(op_type=OPType.NOP)

        if expr and (expr.upper() != "NOP"):
            raise SMT96ASMParseError(f"不支持的操作 {expr}")

        field_value = OPField(op_code=OPCode.NOP, fields=SMT96_FIELD_EMPTY[OPCode.NOP])
        super().__init__(
            op_type=OPType.NOP,
            op_0=OPCode.NOP,
            field_0=field_value,
            op_1=OPCode.NOP,  # 其实无所谓
            field_1=field_value,
        )

    @classmethod
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list[str]:
        """解析表达式

        Example:
            >>> regs = RegisterCollection()
            >>> NOP.parse_expr("NOP", regs)
            ['NOP']
            >>> list(NOP.create_from_expr("NOP; NOP; NOP", regs))[0].asm_value
            'NOP'
            >>> NOP.parse_expr("NOP, R=3", regs)
            Traceback (most recent call last):
            ...
            snn_compiler.backend.smt_96_stmt.smt96.SMT96ASMParseError: 不支持的操作 R=3

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list[str]: 解析后的指令列表, e.g. `["nop", "nop"]`
        """
        result: list[str] = []
        for nop in expr.split(","):
            if (nop := nop.strip().upper()) == "NOP":
                result.append(nop)
                continue
            raise SMT96ASMParseError(f"不支持的操作 {nop}")
        return result

    @property
    def asm_value(self) -> str:
        """汇编语句

        Returns:
            str: 汇编语句
        """
        return "NOP"


__all__ = ["NOP"]
