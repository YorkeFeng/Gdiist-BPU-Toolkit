"""SMT 96 位指令基类"""

from __future__ import annotations

import re
from ctypes import Union
from typing import Iterator

import pyparsing as pp

from ...common import smt_96_base
from ...common.smt_96_base import CTRL_LEVEL, CTRL_PULSE, IEEE754, REG_INDEX, IBinary
from ...common.smt_96_reg import Register96, RegisterCollection
from ..smt_96_op import OPCode, OPField, OPType


class SMT96:
    """96-bit SMT 指令"""

    # region: 成员
    op_type: OPType
    """指令代码, 8 位
    """

    op_0: OPCode
    """操作数 0, 2 位
    """

    field_0: OPField
    """字段 0, 42 位
    """

    op_1: OPCode
    """操作数 1, 2 位
    """

    field_1: OPField
    """字段 1, 42 位
    """

    @property
    def fields(self) -> list:
        """字段列表"""
        return [self.field_0, self.field_1]

    @property
    def ops(self) -> list:
        """操作数列表"""
        return [self.op_0, self.op_1]

    ZERO_REG: Register96 = Register96(index=REG_INDEX.ZERO_REG, name="ZERO_REG")
    """保存 0, 用于无效化一些指令目标寄存器, 与 `RegisterCollection` 的 `ZERO_REG` hash 相同"""

    NONE_REG: Register96 = Register96(index=REG_INDEX.NONE_REG, name="NONE_REG")
    """不存在的寄存器, 用于无效化一些指令目标寄存器, 与 `RegisterCollection` 的 `NONE_REG` hash 相同"""
    # endregion: 成员

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        op_type: OPType = OPType.NOP,
        op_0: OPCode = OPCode.NOP,
        field_0: OPField = None,
        op_1: OPCode = OPCode.NOP,
        field_1: OPField = None,
    ) -> None:
        """构造函数

        Args:
            op_type (OPType, Optional): 指令代码, 默认值 NOP
            op_0 (OPCode, Optional): 操作数 0, 默认值 NOP
            field_0 (OPField, Optional): 字段 0, 默认值 None
            op_1 (OPCode, Optional): 操作数 1, 默认值 NOP
            field_1 (OPField, Optional): 字段 1, 默认值 None
        """
        self.op_type = op_type
        self.op_0 = op_0
        self.field_0 = field_0
        self.op_1 = op_1
        self.field_1 = field_1

    @property
    def bin_value_for_human(self) -> str:
        """二进制数值

        >>> op_type = OPType.NOP
        >>> field_all = OPField(op_code=OPCode.NOP, fields=[])
        >>> SMT96(op_type, OPCode.NOP, field_all, OPCode.NOP, field_all).bin_value_for_human
        '00000000_00_000000000000000000000000000000001111111111_00_000000000000000000000000000000001111111111'

        Returns:
            str: 二进制数值
        """
        result = []
        result.append(IBinary.dec2bin(self.op_type.dec_value, 8))
        result.append(IBinary.dec2bin(self.op_0.dec_value, 2))
        result.append(IBinary.dec2bin(self.field_0.dec_value, 42))
        result.append(IBinary.dec2bin(self.op_1.dec_value, 2))
        result.append(IBinary.dec2bin(self.field_1.dec_value, 42))
        return "_".join(result)

    @property
    def asm_value(self) -> str:
        """汇编语句

        Returns:
            str: 汇编语句
        """
        raise NotImplementedError(f"{self.__class__} 没有 `asm_value` 实现.")

    @classmethod
    def parse_expr(cls, expr: str, regs: RegisterCollection) -> list:
        """解析表达式, 需要子类实现

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Returns:
            list: 解析后的指令列表
        """
        raise NotImplementedError(f"{cls.__name__} 没有 `parse_expr` 实现.")

    @staticmethod
    def get_constant_parser() -> pp.ParserElement:
        """获取常数解析器

        - int 常数数值: 保持原状
        - float 常数数值: 生成 `IEEE754` 对象
        - str 常数名称: 作为整数生成 `IEEE754` 对象

        Example:
            >>> regs = RegisterCollection()
            >>> SMT96.get_constant_parser().parse_string("WRIGHT_RX_READY").as_list()
            [<CTRL_PULSE.WRIGHT_RX_READY: 32>]
            >>> SMT96.get_constant_parser().parse_string("CTRL_PULSE.WRIGHT_RX_READY").as_list()
            [<CTRL_PULSE.WRIGHT_RX_READY: 32>]
            >>> SMT96.get_constant_parser().parse_string("0.4").as_list()
            [<IEEE754:1053609165:0.40>]
            >>> SMT96.get_constant_parser().parse_string("4").as_list()
            [4]

        Returns:
            pp.ParserElement: 常数解析器
        """

        def get_constant(tokens: pp.ParseResults) -> Union[int, IEEE754]:
            """获取常量数值

            - int 常数数值: 保持原状
            - float 常数数值: 生成 `IEEE754` 对象

            Args:
                tokens (ParseResults): Token 列表

            Returns:
                Union[int, IEEE754]: 常量数值
            """
            if "." in tokens[0]:
                return IEEE754(float(tokens[0]))  # 浮点数转化为 IEEE 754 格式

            if tokens[0].isdigit():
                return int(tokens[0])  # 整数保持不变
            raise SMT96ASMParseError(f"不支持的常量 {tokens[0] = }")

        def get_named_constant(tokens: pp.ParseResults) -> Union[CTRL_PULSE, CTRL_LEVEL]:
            """获取命名常数对象,
            e.g. CFG_EN, CTRL_PULSE.WRIGHT_RX_READY

            Args:
                tokens (ParseResults): Token 列表

            Returns:
                Union[CTRL_PULSE, CTRL_LEVEL]: 命名常数对象
            """
            *enum_class, enum_name = tokens[0].rsplit(".", 1)

            if enum_class:
                enum_class = getattr(smt_96_base, enum_class[0], None)
                if enum_class is None:
                    raise SMT96ASMParseError(f"不支持的常量 {tokens[0]}")
                result = getattr(enum_class, enum_name, None)
                if result is None:
                    raise SMT96ASMParseError(f"{enum_class} 没有 {enum_name} 成员")
            else:
                result = getattr(CTRL_PULSE, enum_name, None)
                result = result or getattr(CTRL_LEVEL, enum_name, None)
                if result is None:
                    raise SMT96ASMParseError(f"不支持的常量 {tokens[0]}")
            return result  # 整数默认不转化成 IEEE 754 格式

        # 所有数值常数传换成 IEEE 754 格式
        constant = pp.Word(pp.nums + ".")
        constant.set_parse_action(get_constant)

        # 带名称的常数, e.g. CFG_EN, CTRL_PULSE.WRIGHT_RX_READY
        named_constant = pp.Word(pp.alphanums.upper() + "_.")
        named_constant.set_parse_action(get_named_constant)

        return constant | named_constant

    @staticmethod
    def get_register_parser(regs: RegisterCollection) -> pp.ParserElement:
        """获取寄存器对象解析器

        - 寄存器编号: 解析成 `Register96` 对象
        - 寄存器名称: 解析成 `Register96` 对象

        Example:
            >>> regs = RegisterCollection()
            >>> regs[3].alias = "test"
            >>> SMT96.get_register_parser(regs).parseString("R_test").as_list()
            [<R3(test) = 0, used by: []>]
            >>> SMT96.get_register_parser(regs).parseString("R1").as_list()
            [<R1 = 0, used by: []>]
            >>> SMT96.get_register_parser(regs).parseString("R_NEU_NUMS").as_list()
            [<NEU_NUMS = 0, used by: [-2]>]

        Args:
            regs (RegisterCollection): 寄存器集合

        Returns:
            pp.ParserElement: 寄存器解析器
        """

        def get_register(tokens: pp.ParseResults) -> Register96:
            """获取寄存器对象

            Args:
                tokens (ParseResults): Token 列表

            Returns:
                Register96: 寄存器对象
            """
            reg_index_or_name = tokens[0]
            try:
                return regs[int(reg_index_or_name)]
            except ValueError:
                return regs[reg_index_or_name]
            except Exception as exc:
                raise SMT96ASMParseError(f"不支持的寄存器 {reg_index_or_name = }") from exc

        # 寄存器编号 R*, e.g. R1
        reg_index = pp.Word(pp.nums).set_parse_action(get_register)
        indexed_reg = pp.Suppress("R") + reg_index

        # 寄存器名称 R_*, e.g. R_ZERO_REG
        reg_name = pp.Word(pp.alphanums + "_").set_parse_action(get_register)
        named_reg = pp.Suppress("R_") + reg_name

        # 所有类型寄存器
        return indexed_reg | named_reg

    @staticmethod
    def get_asm_value(value: Union[IEEE754, Register96]) -> str:
        """得到汇编指令中使用的名称

        - `IEEE754` 对象: 返回原始值
        - `Register96` 对象: 返回寄存器编号
        - int: 返回整数
        - 其他: 抛出异常

        Args:
            value (Union[IEEE754, Register96]): 值

        Returns:
            str: 汇编指令中使用的名称
        """
        if isinstance(value, Register96):
            if value.index > 15:
                return f"R_{value.name}"
            return f"R{value.index}"

        if isinstance(value, int):
            return str(value)

        if isinstance(value, IEEE754):
            if isinstance(value.raw_value, int):
                return str(value.raw_value)
            if isinstance(value.raw_value, float):
                return f"{value.raw_value:.2f}"
            if isinstance(value.raw_value, IEEE754):
                return SMT96.get_asm_value(value.raw_value)
            raise ValueError(f"不支持的类型 {value.raw_value = } {type(value.raw_value) = }")

        if isinstance(value, float):
            return str(value)

        raise ValueError(f"不支持的类型 {value = } {type(value) = }")

    @classmethod
    def create_from_expr(cls, expr: str, regs: RegisterCollection) -> Iterator[SMT96]:
        """从表达式加载 `SMT96` 对象

        Args:
            expr (str): 表达式
            regs (RegisterCollection): 寄存器集合

        Yields:
            SMT96: 由表达式生成的 SMT 指令列表
        """
        i = 0
        for one_expr in re.split(r";|\n", expr):
            one_expr = one_expr.split(":", 1)[-1]  # 1: statement # comment
            one_expr = one_expr.split("//", 1)[0]  # 1: statement // comment
            one_expr = one_expr.split("#", 1)[0]  # 1: statement # comment
            if not (one_expr := one_expr.strip()):
                continue
            parsed = False
            for smt96_op in SMT96.__subclasses__():
                try:
                    result = smt96_op(one_expr, regs)
                    parsed = True
                    yield result
                    i += 1
                    break
                except (SMT96ASMParseError, pp.ParseException):
                    continue
            if not parsed:
                raise SMT96ASMParseError(f"无法解析 {one_expr}")

    def __str__(self) -> str:
        return self.bin_value_for_human

    def __repr__(self) -> str:
        return self.asm_value


class SMT96ASMParseError(ValueError):
    """SMT96 汇编语句解析错误"""


__all__ = ["SMT96", "SMT96ASMParseError"]
