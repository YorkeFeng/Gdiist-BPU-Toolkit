"""SMT 96位指令"""

from __future__ import annotations

import re
from enum import auto
from typing import Union

from strenum import UppercaseStrEnum

from ..common.smt_96_reg import Register96

from ..common.smt_96_base import IEEE754, SMT96_FIELD_EMPTY, SMT96_FIELD_FORMAT, IBinary


class OPType(UppercaseStrEnum):
    """5 位指令类型

    - 0: NOP
    - 1: SRAM_OP 数据搬移指令
    - 2: REG_OP 赋值指令
    - 3: FP_OP 算数指令
    - 2: JUMP_OP 跳转指令
    """

    NOP = auto()
    """空操作: 0
    """

    SRAM_OP = auto()
    """SRAM 数据访问: 1

    SRAM 保存与 SRAM 读取

    - SRAM_LOAD: op = 0
    - SRAM_SAVE: op = 1
    """

    REG_OP = auto()
    """赋值: 2

    包括 R, I, B 类型:

    - REG_SET: op = 0: Register, 操作寄存器
    - IMM_SET: op = 1: Immediate, 操作数值
    - BUS_SET: op = 2: Bus, 操作 Bus # FIXME Bus 操作是什么意思?
    """

    FP_OP = auto()
    """计算语句: 3
    """

    JUMP_OP = auto()
    """跳转: 2

    通过赋值 (REG_OP) 寄存器 30 将 SMT_JUMP 设置为 1
    """

    @property
    def dec_value(self) -> int:
        """得到十进制数值

        Returns:
            int: 十进制数值
        """
        return {
            "NOP": 0,
            "SRAM_OP": 1,
            "REG_OP": 2,
            "FP_OP": 3,
            "JUMP_OP": 2,
        }[self.name]

    @property
    def operand(self) -> str:
        """二进制操作数

        >>> OPType.FP_OP.operand
        '00011'

        Returns:
            str: 二进制操作数
        """
        return IBinary.dec2bin(self.dec_value, 5)


class OPCode(UppercaseStrEnum):
    """2 位指令代码

    - NOP: 0

    #### SRAM 操作
    - SRAM_LOAD: 0
    - SRAM_SAVE: 1

    #### 赋值操作
    - REG_SET: 0
    - IMM_SET: 1
    - BUS_SET: 2

    #### 计算操作, 加减法
    - REG_ADD_POS: 0
    - IMM_ADD_POS: 1
    - REG_ADD_NEG: 2
    - IMM_ADD_NEG: 3

    #### 计算操作, 乘法
    - REG_MUL_POS: 0
    - IMM_MUL_POS: 1
    - REG_MUL_NEG: 2
    - IMM_MUL_NEG: 3
    """

    NOP = auto()
    """空操作
    """

    # region: SRAM 操作
    SRAM_LOAD = auto()
    """SRAM 读取
    """

    SRAM_SAVE = auto()
    """SRAM 保存
    """
    # endregion: SRAM 操作

    # region: 赋值操作
    REG_SET = auto()
    """寄存器操作
    """

    IMM_SET = auto()
    """数值操作
    """

    BUS_SET = auto()
    """BUS 操作. # FIXME: Bus 是什么
    """
    # endregion: 赋值操作

    # region: 计算操作
    REG_ADD_POS = auto()
    """寄存器加寄存器
    """

    IMM_ADD_POS = auto()
    """数值加寄存器
    """

    REG_ADD_NEG = auto()
    """寄存器减寄存器
    """

    IMM_ADD_NEG = auto()
    """数值减寄存器
    """

    REG_MUL_POS = auto()
    """寄存器乘寄存器
    """

    IMM_MUL_POS = auto()
    """数值乘寄存器
    """

    REG_MUL_NEG = auto()
    """寄存器乘寄存器的负值
    """

    IMM_MUL_NEG = auto()
    """数值乘寄存器的负值
    """
    # endregion: 计算操作

    @property
    def dec_value(self) -> int:
        """得到数值

        Returns:
            int: 数值
        """
        return {
            "NOP": 0,
            "SRAM_LOAD": 0,
            "SRAM_SAVE": 1,
            "REG_SET": 0,
            "IMM_SET": 1,
            "BUS_SET": 2,
            "REG_ADD_POS": 0,
            "IMM_ADD_POS": 1,
            "REG_ADD_NEG": 2,
            "IMM_ADD_NEG": 3,
            "REG_MUL_POS": 0,
            "IMM_MUL_POS": 1,
            "REG_MUL_NEG": 2,
            "IMM_MUL_NEG": 3,
        }[self.name]

    @property
    def operand(self) -> str:
        """二进制操作数

        >>> OPCode.REG_ADD_NEG.operand
        '10'

        Returns:
            str: 二进制操作数
        """
        return IBinary.dec2bin(self.dec_value, 2)


class OPField(IBinary):
    """42 位操作字段

    - `op_code` (OPCode): 指令代码
    - `field_format` (list[int]): 字段格式
    - `operand` (str): 用于 SMT 96 指令的 42 位二进制操作数
    - `bin_fields` (list[str]): 字段列表, 二进制格式
    - `fields` (list[IEEE754, int]): 字段列表:
        - `int`: 寄存器编号
        - `IEEE754`: 常数
    - `is_empty` (bool): 是否为空字段
    - `empty_field` (OPField): 当前指令代码的空字段

    Example:
        >>> f = OPField(op_code=OPCode.REG_SET, fields=[21, 10, 31, 1])
        >>> f
        [21, 10, 31, 1]
        >>> f.field_format
        [-22, 5, 5, 5, 5]
        >>> f.bin_fields
        ['10101', '01010', '11111', '00001']
        >>> f.fields
        [21, 10, 31, 1]
        >>> f.operand
        '000000000000000000000010101010101111100001'
        >>> f.is_empty
        False
        >>> f.empty_field.is_empty
        True
    """

    op_code: OPCode
    """指令代码"""

    raw_fields: list[Union[IEEE754, int, str, float]]
    """原始字段列表"""

    @property
    def field_format(self) -> list[int]:
        """字段格式, e.g. `"NOP": [-32, 42 + 5, 42 + 5]`

        每个字段的宽度 (width):
        - < 0: 补 `"0" * abs(width)`
        - > 42: 补 `"1" * (width - 42)`

        Returns:
            list[int]: 字段格式
        """
        if self.op_code not in SMT96_FIELD_FORMAT:
            raise NotImplementedError(f"不支持的 {self.op_code.name = }")
        return SMT96_FIELD_FORMAT[self.op_code.name]

    def __init__(
        self,
        op_code: OPCode = OPCode.NOP,
        fields: list[IEEE754, int, float] = None,
    ) -> None:
        """构造函数

        Args:
            op_code (OPCode): 操作代码, 默认为 `OPCode.NOP`
            fields (list[Union[IEEE754, int, str, float]]): 子字段列表
                int: 寄存器编号,
                str: 常数二进制数值,
                IEEE754: 常数浮点数,
                float: 常数浮点数
        """
        super().__init__(dec_value=0, bin_width=42)
        self.op_code = op_code
        if fields is None:
            self.raw_fields = self.bin_fields
        self.fields = fields

    def __str__(self) -> str:
        return str(self.fields)

    def __repr__(self) -> str:
        return str(self.fields)

    @property
    def all_bin_fields(self) -> list[str]:
        """返回一个包括空字段在内的二进制字段列表

        Example:
            >>> f = OPField(op_code=OPCode.REG_SET, fields=[21, 10, 31, 1])
            >>> f.field_format
            [-22, 5, 5, 5, 5]
            >>> f.all_bin_fields
            ['0000000000000000000000', '10101', '01010', '11111', '00001']
            >>> f.bin_fields
            ['10101', '01010', '11111', '00001']

        Returns:
            list[str]: 包括空字段在内的二进制字段列表
        """
        result: list[str] = []
        start = 0
        bin_value = self.bin_value
        for width in self.field_format:
            width = abs(width)
            result.append(bin_value[start : start + width])
            start += width
        return result

    @property
    def bin_fields(self) -> list[str]:
        """返回一个不包括空字段在内的二进制字段列表

        Example:
            >>> f = OPField(op_code=OPCode.REG_SET, fields=[21, 10, 31, 1])
            >>> f.field_format
            [-22, 5, 5, 5, 5]
            >>> f.all_bin_fields
            ['0000000000000000000000', '10101', '01010', '11111', '00001']
            >>> f.bin_fields
            ['10101', '01010', '11111', '00001']

        Returns:
            list[str]: 不包括空字段在内的二进制字段列表
        """
        result: list[str] = []
        for length, bin_value in zip(self.field_format, self.all_bin_fields):
            if 0 < length <= 42:  # 跳过空字段
                result.append(bin_value)
        return result

    @property
    def dec_fields(self) -> list[int]:
        """返回一个不包括空字段在内的整数字段列表

        - 注意: 不能区别常数数值和寄存器编号
        - 请用 `fields` 属性区别常数数值和寄存器编号

        Example:
            >>> f = OPField(op_code=OPCode.IMM_SET, fields=[21.3, 10])
            >>> f.field_format
            [32, 5, 47]
            >>> f.bin_value
            '010000011010101001100110011001100101011111'
            >>> f.bin_fields
            ['01000001101010100110011001100110', '01010']
            >>> f.dec_fields
            [1101686374, 10]
            >>> f.fields
            [<IEEE754:1101686374:21.30>, 10]

        Returns:
            list[IEEE754, int]: 包含十进制字段的列表
        """
        return [IBinary.bin2dec(bin_value) for bin_value in self.bin_fields]

    @property
    def fields(self) -> list[int]:
        """字段列表, 不包括空字段

        - int: 寄存器编号
        - IEEE754: 常数浮点数

        Example:
            >>> f = OPField(op_code=OPCode.IMM_SET, fields=[21.3, 10])
            >>> f.field_format
            [32, 5, 47]
            >>> f.fields
            [<IEEE754:1101686374:21.30>, 10]
            >>> f.fields = [21, 10]
            >>> f.fields
            [<IEEE754:21:0.00>, 10]

        Returns:
            list[IEEE754, int]: 字段列表
        """
        result: list[Union[IEEE754, int]] = []
        for bin_value in self.bin_fields:
            if len(bin_value) == 32:  # REVIEW: 32 位一定是 IEEE754 浮点数
                result.append(IEEE754(bin_value))
                continue
            result.append(IBinary.bin2dec(bin_value))
        return result

    @fields.setter
    # pylint: disable-next=too-many-branches
    def fields(self, value: list[Union[IEEE754, int, str, float]]) -> None:
        """设置字段各个子字段, 保存字段的原始数据

        - 如果 `value` 是 None, 则不做任何操作
        - 更新 `self.bin_value` 和 `self.raw_fields`

        Raises:
            ValueError: 如果字段长度不对

        Args:
            value (list[Union[IEEE754, int, str, float]]): 子字段列表
                int: 寄存器编号,
                str: 常数二进制数值,
                IEEE754: 常数浮点数,
                float: 常数浮点数
        """
        if value is None:
            return

        # self.fields 一直存在, 是根据 self.bin_value 计算的
        if len(value) != len(self.fields):
            raise ValueError(f"字段长度不对: {len(value)} != {len(self.fields)}")

        bin_fields = ["" for _ in self.field_format]
        idx = 0
        for i, length in enumerate(self.field_format):
            if length < 0:  # 用 "0" 填充空字段
                bin_fields[i] = "0" * abs(length)
                continue
            if length > 42:  # 用 "1" 填充空字段
                bin_fields[i] = "1" * (length - 42)
                continue

            field = value[idx]
            idx += 1

            if isinstance(field, IEEE754):  # IEEE754 浮点数, length 必须是 32
                if length != 32:
                    raise ValueError(f"数值是 IEEE754, 字段长度必须是 32, {field = }, {length = }")
                bin_value = field.bin_value
            elif isinstance(field, str):  # 二进制数值, length 可以不是 32
                field = re.sub("^0b", "", field)
                if len(field) != length:
                    raise ValueError(f"二进制 {field = } 长度不对, 应该是 {length}")
                bin_value = field
            elif isinstance(field, int):  # 整数, length 可以不是 32
                if field < 0:
                    raise ValueError(f"整数 (寄存器编号) {field = } 必须是正数")
                bin_value = IBinary.dec2bin(field, length)
            elif isinstance(field, float):  # 浮点数, length 必须是 32
                if length != 32:
                    raise ValueError(f"数值是 float, 字段长度必须是 32, {field = }, {length = }")
                bin_value = IEEE754(field).bin_value
            elif isinstance(field, Register96):  # 寄存器对象 length 必须是 5
                if length != 5:
                    raise ValueError(f"数值是 Register96, 字段长度必须是 5, {field = }, {length = }")
                bin_value = IBinary.dec2bin(field.index, 5)
            else:
                raise TypeError(f"{field = } ({type(field)}) 类型不对, 只能是 str, int, float, IEEE754 或者 Register96")

            bin_fields[i] = bin_value
        self.bin_value = "".join(bin_fields)
        self.raw_fields = value

    def update_field(self, value: Union[IEEE754, int, str, float], index: int) -> None:
        """更新字段

        Example:
            >>> f = OPField(op_code=OPCode.REG_SET)
            >>> f.bin_fields
            ['00000', '00000', '00000', '00000']
            >>> f.update_field(3, 1)
            >>> f
            [0, 3, 0, 0]
            >>> f.raw_fields
            ['00000', 3, '00000', '00000']

        Args:
            value (Union[IEEE754, int, str, float]): 值
            index (int): 字段编号
        """
        fields = self.raw_fields.copy()
        fields[index] = value
        self.fields = fields

    @property
    def operand(self) -> str:
        """`self.bin_value` 的别名

        Example:
            >>> f = OPField(op_code=OPCode.IMM_SET, fields=[21.3, 10])
            >>> f.field_format
            [32, 5, 47]
            >>> f.fields
            [<IEEE754:1101686374:21.30>, 10]
            >>> f.bin_value
            '010000011010101001100110011001100101011111'
            >>> f.operand
            '010000011010101001100110011001100101011111'

        Returns:
            str: 二进制操作数
        """
        return self.bin_value

    @property
    def is_empty(self) -> bool:
        """是否为空字段

        - `self.op_code == OPCode.NOP` 总是返回 True
        - 其他指令: `self.dec_fields == SMT96_FIELD_EMPTY[self.op_code]`


        Example:
            >>> f = OPField(OPCode.NOP)
            >>> f
            []
            >>> f.is_empty
            True
            >>> f = OPField(OPCode.SRAM_LOAD)
            >>> f.bin_value = "000000000000000000000010101010101111100001"
            >>> f
            [21, 10, 31, 1]
            >>> f.is_empty
            False
            >>> f = OPField(OPCode.SRAM_SAVE, fields=[16, 16, 16, 16])
            >>> f.is_empty
            True

        Returns:
            bool: 是否为空字段
        """
        if self.op_code == OPCode.NOP:
            return True

        if self.op_code.name not in SMT96_FIELD_EMPTY:
            raise NotImplementedError(f"不支持空字段判断: {self.op_code.name = }")

        empty_fields = SMT96_FIELD_EMPTY[self.op_code]

        for dec_value, empty_value in zip(self.dec_fields, empty_fields):
            if dec_value != empty_value:
                return False

        return True

    @property
    def empty_field(self) -> OPField:
        """空字段对象, 通过 `SMT96_FIELD_EMPTY[self.op_code]` 得到

        Example:
            >>> f = OPField(op_code=OPCode.REG_SET, fields=[21, 10, 31, 1])
            >>> f.field_format
            [-22, 5, 5, 5, 5]
            >>> f.empty_field
            [16, 16, 31, 31]
            >>> f.empty_field.is_empty
            True

        Returns:
            OPField: 空字段对象
        """
        result = OPField(op_code=self.op_code, fields=SMT96_FIELD_EMPTY[self.op_code])
        return result
