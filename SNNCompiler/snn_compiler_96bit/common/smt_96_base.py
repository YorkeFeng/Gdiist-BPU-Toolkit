"""这个模块包含了用于处理 SMT 96 位指令的常用函数和类
"""

from __future__ import annotations

import re
import struct
from ctypes import Union
from dataclasses import dataclass
from enum import IntEnum


# region: 常数
class CTRL_LEVEL(IntEnum):
    """NPU 控制信号

    - `CFG_EN`: `0b0000_0001`, NPU 配置必要的寄存器
    - `NPU_RST_EN`: `0b0000_0010`, NPU 运行完一个 step 或者 cfg 完成, 等待 run 拉高开始下个 step 的计算
    - `SIM_EN`: `0b0000_0100`, NPU 仿真开始
    - `W_EN`: `0b0000_1000`, NPU 进入权重累加阶段
    - `V_EN`: `0b0001_0000`, NPU 进入膜电位更新阶段
    - `S_EN`: `0b0010_0000`, step += 1
    - `NDMA_RD_EN`: `0b0100_0000`, NPU发控制信号给 ndma, 从 weight sram 读数据到 NPU sram
    - `NDMA_WR_EN`: `0b1000_0000`, NPU发控制信号给 ndma, 将 NPU sram 的数据回写到 weight sram

    """

    CFG_EN: int = 1 << 0
    """NPU 配置必要的寄存器"""

    NPU_RST_EN: int = 1 << 1
    """NPU 运行完一个 step 或者 cfg 完成, 等待 run 拉高开始下个 step 的计算"""

    SIM_EN: int = 1 << 2
    """NPU 仿真开始"""

    W_EN: int = 1 << 3
    """NPU 进入权重累加阶段"""

    V_EN: int = 1 << 4
    """NPU 进入膜电位更新阶段"""

    S_EN: int = 1 << 5
    """step += 1"""

    NDMA_RD_EN: int = 1 << 6
    """NPU发控制信号给 ndma, 从 weight sram 读数据到 NPU sram"""

    NDMA_WR_EN: int = 1 << 7
    """NPU发控制信号给 ndma, 将 NPU sram 的数据回写到 weight sram"""


class REG_INDEX(IntEnum):
    """寄存器索引, 16-31"""

    ZERO_REG = 16
    """保存 0, 用于无效化一些指令目标寄存器"""

    STEP_REG = 17
    """保存当前步数"""

    NEU_NUMS = 18
    """保存神经元数量"""

    CHIP_NPU_ID = 19
    """保存 NPU ID"""

    TRST_REG0 = 20
    """保存 T_RST 信号"""

    TRST_REG1 = 21
    """保存 T_RST 信号"""

    VRST_REG0 = 22
    """保存 V_RST 信号"""

    VRST_REG1 = 23
    """保存 V_RST 信号"""

    PHASE = 24
    """保存相位信息"""

    V_DIFF_REG0 = 25
    """保存 V_DIFF 信号"""

    V_DIFF_REG1 = 26
    """保存 V_DIFF 信号"""

    CTRL_LEVEL = 27
    """NPU 控制寄存器"""

    TLASTSP_TMP0 = 28
    """保存 T_LASTSP 信号"""

    TLASTSP_TMP1 = 29
    """保存 T_LASTSP 信号"""

    CTRL_PULSE = 30
    """保存控制脉冲"""

    NONE_REG = 31
    """不存在的寄存器, 用于无效化一些指令目标寄存器"""


class CTRL_PULSE(IntEnum):
    """NPU 控制脉冲, 写入 R30 (CTRL_PULSE)

    - `TIMER_SET = 1 << 0`
    - `SMT_JUMP = 1 << 1`
    - `SIM_END = 1 << 2`
    - `NDMA_RD = 1 << 3`
    - `NDMA_WR = 1 << 4`
    - `WRIGHT_RX_READY = 1 << 5`
    - `TRANS_SEL = 1 << 6`
    - `STEP_PLUS = 1 << 7`
    - `STEP_NPU_SET = 1 << 8`
    - `V_JUMP = 1 << 9`
    - `NPU_SET = 1 << 10`
    - `NPU_PLUS = 1 << 11`
    - `SPIKE = 1 << 12`
    - `T_SET = 1 << 13`
    - `T_UPDATE = 1 << 14`
    - `W_JUMP = 1 << 15`
    - `V_SET = 7 << 16`
    - `NEU_ID = 1 << 19`
    - `LFSR_INIT_SET = 1 << 20`
    - `LFSR_SET = 1 << 21`

    >>> CTRL_PULSE["TIMER_SET"]
    <CTRL_PULSE.TIMER_SET: 1>
    """

    TIMER_SET = 1 << 0
    """`1 << 0`"""

    SMT_JUMP = 1 << 1
    """`1 << 1`"""

    SIM_END = 1 << 2
    """`1 << 2`"""

    NDMA_RD = 1 << 3
    """`1 << 3`"""

    NDMA_WR = 1 << 4
    """`1 << 4`"""

    WRIGHT_RX_READY = 1 << 5
    """`1 << 5`"""

    TRANS_SEL = 1 << 6
    """`1 << 6`"""

    STEP_PLUS = 1 << 7
    """`1 << 7`"""

    STEP_NPU_SET = 1 << 8
    """`1 << 8`"""

    V_JUMP = 1 << 9
    """`1 << 9`"""

    NPU_SET = 1 << 10
    """`1 << 10`"""

    NPU_PLUS = 1 << 11
    """`1 << 11`"""

    SPIKE = 1 << 12
    """`1 << 12`"""

    T_SET = 1 << 13
    """`1 << 13`"""

    T_UPDATE = 1 << 14
    """`1 << 14`"""

    W_JUMP = 1 << 15
    """`1 << 15`"""

    V_SET = 1 << 16
    """`1 << 16`"""

    NEU_ID = 1 << 19
    """`1 << 19`"""

    LFSR_INIT_SET = 1 << 20
    """`1 << 20`"""

    LFSR_SET = 1 << 21
    """`1 << 21`"""


SMT96_FIELD_FORMAT = {
    "NOP": [-32, 42 + 5, 42 + 5],
    "SRAM_LOAD": [-22, 5, 5, 5, 5],
    "SRAM_SAVE": [-22, 5, 5, 5, 5],
    "REG_SET": [-22, 5, 5, 5, 5],
    "IMM_SET": [32, 5, 42 + 5],
    "BUS_SET": [-22, 5, 5, 5, 5],
    "REG_ADD_POS": [-27, 5, 5, 5],
    "IMM_ADD_POS": [32, 5, 5],
    "REG_ADD_NEG": [-27, 5, 5, 5],
    "IMM_ADD_NEG": [32, 5, 5],
    "REG_MUL_POS": [-27, 5, 5, 5],
    "IMM_MUL_POS": [32, 5, 5],
    "REG_MUL_NEG": [-27, 5, 5, 5],
    "IMM_MUL_NEG": [32, 5, 5],
}
"""SMT 96-bit 指令 field 格式
```python
{
    "NOP": [-32, 42 + 5, 42 + 5],
    "SRAM_LOAD": [-22, 5, 5, 5, 5],
    "SRAM_SAVE": [-22, 5, 5, 5, 5],
    "REG_SET": [-22, 5, 5, 5, 5],
    "IMM_SET": [32, 5, 42 + 5],
    "BUS_SET": [-22, 5, 5, 5, 5],
    "REG_ADD_POS": [-27, 5, 5, 5],
    "IMM_ADD_POS": [32, 5, 5],
    "REG_ADD_NEG": [-27, 5, 5, 5],
    "IMM_ADD_NEG": [32, 5, 5],
    "REG_MUL_POS": [-27, 5, 5, 5],
    "IMM_MUL_POS": [32, 5, 5],
    "REG_MUL_NEG": [-27, 5, 5, 5],
    "IMM_MUL_NEG": [32, 5, 5],
}
```
"""


SMT96_FIELD_EMPTY = {
    "NOP": [],
    "SRAM_LOAD": [
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
    ],  # 读取到不能达到的 16
    "SRAM_SAVE": [
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
    ],  # 从不能达到的 16 保存
    "REG_SET": [
        REG_INDEX.NONE_REG, # uger debug, ori: REG_INDEX.ZERO_REG
        REG_INDEX.NONE_REG, # uger debug
        REG_INDEX.NONE_REG,
        REG_INDEX.NONE_REG,
    ],  # REG_INDEX.NONE_REG = REG_INDEX.ZERO_REG
    "IMM_SET": [0.0, REG_INDEX.NONE_REG],  # REG_INDEX.NONE_REG = 0.0
    "BUS_SET": [
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.NONE_REG,
        REG_INDEX.NONE_REG,
    ],  # REG_INDEX.NONE_REG = REG_INDEX.ZERO_REG
    "REG_ADD_POS": [
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.NONE_REG,
    ],  # REG_INDEX.NONE_REG = REG_INDEX.ZERO_REG + REG_INDEX.ZERO_REG
    "IMM_ADD_POS": [0.0, REG_INDEX.ZERO_REG, REG_INDEX.NONE_REG],  # REG_INDEX.NONE_REG = 0.0 + REG_INDEX.ZERO_REG
    "REG_ADD_NEG": [
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.NONE_REG,
    ],  # REG_INDEX.NONE_REG = REG_INDEX.ZERO_REG - REG_INDEX.ZERO_REG
    "IMM_ADD_NEG": [0.0, REG_INDEX.ZERO_REG, REG_INDEX.NONE_REG],  # REG_INDEX.NONE_REG = 0.0 - REG_INDEX.ZERO_REG
    "REG_MUL_POS": [
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.NONE_REG,
    ],  # REG_INDEX.NONE_REG = REG_INDEX.ZERO_REG * REG_INDEX.ZERO_REG
    "IMM_MUL_POS": [0.0, REG_INDEX.ZERO_REG, REG_INDEX.NONE_REG],  # REG_INDEX.NONE_REG = 0.0 * REG_INDEX.ZERO_REG
    "REG_MUL_NEG": [
        REG_INDEX.ZERO_REG,
        REG_INDEX.ZERO_REG,
        REG_INDEX.NONE_REG,
    ],  # REG_INDEX.NONE_REG = REG_INDEX.ZERO_REG * (-REG_INDEX.ZERO_REG)
    "IMM_MUL_NEG": [0.0, REG_INDEX.ZERO_REG, REG_INDEX.NONE_REG],  # REG_INDEX.NONE_REG = 0.0 * (-REG_INDEX.ZERO_REG)
}
"""SMT 96-bit 指令空 field 参数. 跳过了填充的 "0" 或者 "1"
"""
# endregion: 常数


@dataclass
class IBinary:
    """二进制数值接口, 无符号

    - `dec_value` (int): 十进制数值, 默认值 0
    - `bin_width` (int): 二进制宽度, 默认值 32
    - `bin_value` (str): 二进制数值字符串, e.g. "000100"
    - `dec2bin` (int, int) -> str: 十进制数值转换成二进制数值
    - `bin2dec` (str, bool) -> int: 二进制数值转换成十进制数值
    """

    dec_value: int = 0
    """十进制数值
    """

    bin_width: int = 32
    """二进制宽度
    """

    def __int__(self) -> int:
        """返回十进制数值"""
        return self.dec_value

    def __post_init__(self) -> None:
        """构造函数"""

        if isinstance(self.dec_value, str):
            self.dec_value = self.bin2dec(self.dec_value)
            return

        if (is_int := isinstance(self.dec_value, int)) and (self.dec_value < 0):
            raise TypeError(f"dec_value 必须是正数 ({self.dec_value = })")

        if is_int and (self.dec_value >= (1 << self.bin_width)):
            raise ValueError(f"{self.dec_value = } 太大. 位宽 {self.bin_width = } 不够")

    @staticmethod
    def dec2bin(dec_value: int, bin_width: int = 32) -> str:
        """十进制数值转换成二进制数值

        Example:
            >>> IBinary.dec2bin(-31, 32)
            Traceback (most recent call last):
            ...
            TypeError: dec_value 必须是正数 (dec_value = -31)
            >>> IBinary.dec2bin(32, 8)
            '00100000'
            >>> IBinary.dec2bin(32, 5)
            Traceback (most recent call last):
            ..
            ValueError: dec_value = 32 太大. 位宽 bin_width = 5 不够
            >>> IBinary.dec2bin(16, 5)
            '10000'

        Args:
            dec_value (int): 十进制数值
            bin_width (int, optional): 二进制数值的宽度. 默认值 32

        Raises:
            TypeError: 如果输入是负数
            ValueError: 如果输入数值超过位宽的范围

        Returns:
            str: 二进制数值
        """
        if dec_value < 0:
            raise TypeError(f"dec_value 必须是正数 ({dec_value = })")

        if dec_value >= (1 << bin_width):
            raise ValueError(f"{dec_value = } 太大. 位宽 {bin_width = } 不够")

        return f"{dec_value:0{bin_width}b}"

    @staticmethod
    def bin2dec(bin_value: str) -> int:
        """二进制数值转换成十进制数值, 无符号

        Example:
            >>> IBinary.bin2dec("0b00000000000000000000000000011111")
            31
            >>> IBinary.bin2dec("100001")
            33
            >>> IBinary.bin2dec("11")
            3

        Args:
            bin_value (str): 二进制数值

        Returns:
            int: 十进制数值
        """
        return int(bin_value, 2)

    @property
    def bin_value(self) -> str:
        """二进制数值, 将十进制数值转换为二进制数值

        Example:
            >>> IBinary(31).bin_value
            '00000000000000000000000000011111'
            >>> IBinary(39).bin_value
            '00000000000000000000000000100111'
            >>> IBinary(31, bin_width=8).bin_value
            '00011111'
            >>> IBinary(33, bin_width=8).bin_value
            '00100001'
            >>> f = IBinary()
            >>> f.bin_value = "0b00000000000000000000000000011111"
            >>> f.dec_value
            31
            >>> f.bin_value = "100001"
            >>> f.dec_value
            33

        Returns:
            str: 转换后的二进制数值
        """
        return self.dec2bin(self.dec_value, self.bin_width)

    @bin_value.setter
    def bin_value(self, value: str) -> None:
        """更新二进制数值, 同时更新十进制数值

        Example:
            >>> f = IBinary()
            >>> f.bin_value = "0b00000000000000000000000000011111"
            >>> f.dec_value
            31
            >>> f.bin_value = "100001"
            >>> f.dec_value
            33

        Args:
            bin_value(str): 二进制数值
            signed(bool, optional): 二进制数值首位是否是符号位. 默认值 False
        """
        self.dec_value = self.bin2dec(value)

    def __str__(self) -> str:
        return str(self.dec_value)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}:{self.dec_value}:{self.bin_value}>"


class IEEE754(IBinary):
    """IEEE 754 格式的 32 位浮点数

    - 整数不转化为 IEEE 754 格式的整数
    - 浮点数转化为 IEEE 754 格式的整数
    - 二进制数转化为整数, 不转化为 IEEE 754 格式的整数

    Example:
        >>> IEEE754(1).dec_value
        1
        >>> IEEE754("101").dec_value
        5
        >>> IEEE754(-1).dec_value
        Traceback (most recent call last):
        ...
        ValueError: value = -1 不是正数
        >>> IEEE754(1.0).dec_value
        1065353216
        >>> IEEE754(-1.0).dec_value
        3212836864
        >>> IEEE754("00111111100000000000000000000000").dec_value
        1065353216
        >>> IEEE754.float_to_ieee754(1.0)
        1065353216
        >>> IEEE754.float_to_ieee754(1)
        1065353216
        >>> IEEE754.ieee754_to_float(1065353216)
        1.0
        >>> -IEEE754(1.0)
        <IEEE754:3212836864:-1.00>
        >>> -(-IEEE754(1.0))
        <IEEE754:1065353216:1.00>
        >>> (-IEEE754(1.0)).float_value
        -1.0
        >>> -IEEE754(1)
        <IEEE754:2147483649:-0.00>
        >>> IEEE754(1).float_value
        1.401298464324817e-45
        >>> (-IEEE754(1)).float_value
        -1.401298464324817e-45
    """

    raw_value: Union[int, float, "IEEE754"]
    """原始值, 二进制字符串按照整数处理"""

    def __init__(self, value: Union[str, int, float, "IEEE754"]) -> None:
        """构造函数

        - 整数不转化为 IEEE 754 格式的整数
        - 浮点数转化为 IEEE 754 格式的整数
        - 二进制数转化为整数, 不转化为 IEEE 754 格式的整数

        Args:
            value (Union[str, int, float, IEEE754]): 整数, 浮点数, 二进制数, IEEE 754 格式的整数
                - IEEE754: IEEE 754 格式的浮点数
                - int: 整数, 不转化为 IEEE 754 格式的整数, 不能是负数
                - float: 浮点数, 转化为 IEEE 754 格式的整数
                - str: 无符号二进制数, 不转化为 IEEE 754 格式的整数
        """
        self.raw_value = value
        if isinstance(value, IEEE754):  # IEEE754 浮点数
            dec_value = value.dec_value
        elif isinstance(value, int):  # 整数, 不转化为 IEEE 754 格式, 不能是负数
            if value < 0:
                raise ValueError(f"{value = } 不是正数")
            dec_value = value
        elif isinstance(value, float):  # 浮点数需要转换成整数
            dec_value = self.float_to_ieee754(value)
        elif isinstance(value, str):  # 二进制需要转换成整数
            if not re.match(r"(0b)?[01]+", value):
                raise ValueError(f"{value = } 不是二进制数值")
            dec_value = IBinary.bin2dec(value)
            self.raw_value = int(value, 2)
        else:
            raise TypeError(f"{value = } 不是 IEEE754, int, float 或 str 类型")

        super().__init__(dec_value=dec_value, bin_width=32)

    @property
    def float_value(self) -> float:
        """浮点数值"""
        return self.ieee754_to_float(self.dec_value)

    @float_value.setter
    def float_value(self, value: Union[int, float]) -> float:
        """改变浮点数值, 同时改变原始值"""
        self.raw_value = value
        self.dec_value = self.float_to_ieee754(value)

    @staticmethod
    def float_to_ieee754(x: Union[int, float]) -> int:
        """将整数或浮点数转换为 IEEE 754 格式的 32 位整数

        Example:
            >>> IEEE754.float_to_ieee754(1.0)
            1065353216

        Args:
            x (Union[int, float]): 整数或浮点数

        Returns:
            int: 整数
        """
        return struct.unpack("<I", struct.pack("<f", x))[0]

    @staticmethod
    def ieee754_to_float(x: int) -> float:
        """将 IEEE 754 格式的整数转换为浮点数

        Example:
            >>> IEEE754.ieee754_to_float(1065353216)
            1.0

        Args:
            x (int): IEEE 754 格式的整数

        Returns:
            float: 浮点数
        """

        return struct.unpack("<f", struct.pack("<I", x))[0]

    def __neg__(self) -> IEEE754:
        """取负值, 原始值也取负值

        Example:
            >>> -IEEE754(1.0)
            <IEEE754:3212836864:-1.00>
            >>> -IEEE754(1.0).raw_value
            -1.0
            >>> -IEEE754(3)
            <IEEE754:2147483651:-0.00>
            >>> -IEEE754(3).raw_value
            -3
            >>> IEEE754(3)
            <IEEE754:3:0.00>
            >>> IEEE754(3).raw_value
            3

        Returns:
            IEEE754: IEEE 754 格式的负值
        """
        self.raw_value = -self.raw_value  # 最终的 IEEE754 对象原始值肯定不是 IEEE754 对象
        return IEEE754(-self.ieee754_to_float(self.dec_value))

    def __repr__(self) -> str:
        """返回字符串

        Example:
            >>> IEEE754(1.0)
            <IEEE754:1065353216:1.00>
            >>> IEEE754(1)
            <IEEE754:1:0.00>
        """
        return f"<{self.__class__.__name__}:{self.dec_value}:{self.float_value:.2f}>"


__all__ = [
    "CTRL_LEVEL",
    "REG_INDEX",
    "SMT96_FIELD_FORMAT",
    "SMT96_FIELD_EMPTY",
    "IBinary",
    "IEEE754",
]
