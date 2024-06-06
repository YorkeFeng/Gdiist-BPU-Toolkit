# pylint: disable=too-few-public-methods, protected-access, invalid-name

"""SMT instruction set and behavior
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any, Union

import numpy as np
from addict import Dict as AttrDict

Number = Union[int, float]


class INumber:
    """数字接口, 可返回:

    - bin_value: 二进制数值, e.g. 0b0111
    - dec_value: 十进制数值, e.g. 7
    - hex_value: 十六进制数值, e.g. 0xC
    """

    value: int = 0
    """十进制数值, 默认为 0
    """

    width: int = 32
    """二进制数值位数, `32`
    """

    def __str__(self) -> str:
        """重载 `self.__str__` 返回二进制数值

        Returns:
            str: 二进制数值
        """
        return self.bin_value

    @property
    def dec_value(self) -> str:
        """返回十进制值, e.g. 7"""
        return str(self._numpy_value)

    @property
    def _numpy_value(self) -> np.float32 | np.int32:
        """返回 numpy 的数据类型

        Raises:
            RuntimeError: 只支持 python 类型 `int` 和 `float`.

        Returns:
            np.float32 | np.int32: numpy 的数据类型
        """
        if isinstance(self.value, int):
            result = np.int32(self.value)
        elif isinstance(self.value, float):
            result = np.float32(self.value)
        else:
            raise RuntimeError(f"不支持 {self.value = }, 类型 {type(self.value)}")

        result = result.astype("float32")
        result = result.view("<u4")
        return result

    @property
    def bin_value(self) -> str:
        """返回二进制值, e.g. 0b011"""

        result = np.base_repr(self._numpy_value, base=2)
        result = result.rjust(self.width, "0")
        result = f"0b{result}"
        return result

    @property
    def hex_value(self) -> str:
        """返回带下划线的十六进制值, e.g. 0x7F"""
        result = np.base_repr(self._numpy_value, base=16)
        result = result.rjust(int(self.width / 4), "0")
        result = f"0x{result}"
        return result


@dataclass
class RegisterBase:
    """32 位寄存器基类"""

    index: int = -1
    """Register index. Defaults to -1 means last Register index + 1.
    """

    used_by: set[int] = field(default_factory=set)
    """使用此寄存器的 IR 语句:

    - int: 语句的索引值, -1 表示函数参数, -2 表示保留寄存器 0 和 1: `ZERO_REG` 和 `ONE_REG`
    """

    alias: str = ""
    """SMT 语句中的寄存器别名, e.g. `V_reset`.
    """

    as_arg: str = ""
    """函数输入的名字, e.g. `I`."""

    as_return: str = ""
    """函数输出的名字, e.g. `I`."""

    _name: str = ""
    """汇编语言中的寄存器名称, e.g. R0. 只读属性."""

    @property
    def name(self) -> str:
        """汇编语言中的寄存器名称, e.g. R0."""
        return self._name

    def update_name(self, value: str) -> None:
        """更新 `self._name`.

        Args:
            value (str): 名字.
        """
        self._name = value

    @property
    def short(self) -> str:
        """寄存器的别名或者名称

        Returns:
            str: 寄存器的别名或者名称
        """
        if self.as_arg:
            return self.as_arg
        if self.alias:
            return self.alias
        return self.name

    value: int = 0
    """当前寄存器的值. Defaults to 0.
    """

    def update(self, **kwargs: Any) -> Register:
        """改变成员的值.

        Args:
            kwargs (Any): 改变寄存器成员的值.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"{self.__class__.__name__} 没有 {k} 成员")
            if k in ["name"]:
                self.update_name(v)
            else:
                setattr(self, k, v)
        return self


class Register(RegisterBase):
    """32 位寄存器"""

    def __hash__(self) -> int:
        """取得哈希码 这样寄存器就可以放到 set 里边了

        Returns:
            int: 哈希码
        """
        return hash(f"{self.name}, {self.index}")

    def release(self) -> None:
        """释放寄存器.

        - `Register.alias = ""`
        - `Register.value = 0`
        - `Register.used_by = set()`
        - `Register.as_arg = ""`
        - `Register.as_return = ""`

        """
        self.alias = ""
        self.value = 0
        self.used_by = set()
        self.as_arg = ""  # 不用做函数输入
        self.as_return = ""  # 不用做函数输出

    def __str__(self) -> str:
        result = self.name
        if self.alias:
            result += f"({self.alias})"
        result += f" = {self.value}, used by: {self.used_by}"
        if self.as_arg:
            result += f", func_arg: {self.as_arg}"
        if self.as_return:
            result += f", func_return: {self.as_return}"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def replace_by(self, reg: Register) -> Register:
        """用 `reg` 的信息替换当前寄存器的信息.

        Args:
            reg (Register): 另一个寄存器

        Returns:
            Register: 修改过的当前寄存器.
        """
        self._name = reg._name
        self.alias = reg.alias
        self.index = reg.index
        self.value = reg.value
        return self

    @property
    def used_by_list(self) -> list[int]:
        """返回 `Register.used_by` 列表.

        Returns:
            list[int]: `Register.used_by` 列表.
        """
        return sorted(list(self.used_by))

    @property
    def first(self) -> int:
        """返回 `Register.used_by` 最小值.
        如果没有 `Register.used_by`, 返回 -1.

        Returns:
            int: `Register.used_by` 最小值.
        """
        if not self.used_by:
            return -1
        return min(self.used_by)

    @property
    def last(self) -> int:
        """返回 `Register.used_by` 最大值.
        如果没有 `Register.used_by`, 返回 -1.

        Returns:
            int: `Register.used_by` 最大值.
        """
        if not self.used_by:
            return -1
        return max(self.used_by)


class Operator(Enum):
    """6 位控制字段"""

    NOP: int = 0
    """6-bit 空操作, 代码 0: 0b000000
    """

    CALCU: int = 1
    """6-bit 计算, 代码 1: 0b000001

    - 32-bit 多路并行加法单元的两个源操作数和一个目的操作数选择: A1 A2 S
    - 32-bit 多路并行乘法单元的两个源操作数和一个目的操作数选择: M1 M2 P
    - 一条 Calcu 指令可以同时执行一次多路并行加法和一次多路并行乘法.
    """

    SRAM_LOAD: int = 2
    """6-bit 读存储, 代码 2: 0b000010

    将存储在 BRAM 缓存中的目的神经元的独享参数载入到独享常量/独享变量寄存器中
    """

    SRAM_SAVE: int = 3
    """6-bit 写存储, 代码 3: 0b000011

    将独享常量/独享变量寄存器中的神经元独享参数存回到 BRAM 缓存中
    """

    V_SET: int = 4
    """6-bit 膜电位更新, 代码 4: 0b000100

    令通过判断源操作数 A1 的值以及内部相关逻辑 (神经元是否处于不应期)
    来决定是否将源操作数 A2 的值赋给目的操作数
    该指令一般用来做膜电位更新操作, 其中 A1 和 A2 为源操作数选择信号, S 为目的操作数选择信号

    Example:

    `V_SET: A1=temp0, A2=V_reset, S=V`

    根据 temp0 的值的最高位是否为负以及是否处于不应期(硬件电路实现), 判断是否将 V_reset 的值赋值给 V.
    """

    SPIKE: int = 5
    """6-bit 脉冲发放, 代码 5: 0b000101

    通过判断源操作数 A1 的值以及内部相关逻辑 (神经元是否处于不应期) 来决定是否发放脉冲
    """

    END: int = 6
    """6-bit 仿真结束, 代码 6: 0b000110

    标志当前计算结束, 令 SMT 地址跳转到首地址开始下一次计算, 控制 SMT 存储器的地址寄存器的更新
    """

    def __str__(self) -> str:
        """输出 `f"{self.name}({self.value})"`

        Returns:
            str: `f"{self.name}({self.value})"`
        """
        return f"{self.name}({self.value})"

    def __repr__(self) -> str:
        """输出 `f"{self.name}({self.value})"`

        Returns:
            str: `f"{self.name}({self.value})"`
        """
        return self.__str__()


@dataclass
class RegisterCollectionBase:  # pylint: disable=too-many-instance-attributes
    """寄存器集合基类"""

    # region: 预设寄存器成员
    R0: Register = None
    """本地寄存器. 不可以用作输入, 可以用作结果寄存器.
    """

    R1: Register = None
    """本地寄存器. 不可以用作输入, 可以用作结果寄存器.
    """

    R2: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R3: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R4: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R5: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R6: Register = None
    """本地寄存器. 可以用作输入, 可以用作结果寄存器.
    """

    R7: Register = None
    """本地寄存器. 可以用作输入, 不可以用作结果寄存器.
    """

    R8: Register = None
    """本地寄存器. 可以用作输入, 不可以用作结果寄存器.
    """

    R9: Register = None
    """本地寄存器. 可以用作输入, 不可以用作结果寄存器.
    """

    R5_NEG: Register = None
    """本地寄存器. 存储 R5 的负值. 不可以用作输入, 不可以用作结果寄存器.
    """

    R6_NEG: Register = None
    """本地寄存器. 存储 R6 的负值. 不可以用作输入, 不可以用作结果寄存器.
    """

    ADD_S: Register = None
    """加法结果本地寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    MUL_P: Register = None
    """乘加法结果本地寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR0: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR1: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR2: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR3: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR4: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR5: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR6: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR7: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR8: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR9: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR10: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR11: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR12: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR13: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR14: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR15: Register = None
    """共享寄存器. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR16: Register = None
    """共享寄存器. 存储常数 1. 不可以用作输入, 不可以用作结果寄存器.
    """

    SR17: Register = None
    """共享寄存器. 存储常数 0. 不可以用作输入, 不可以用作结果寄存器.
    """
    # endregion: 预设寄存器成员

    all_registers: dict[int, Register] = None
    """所有寄存器.
    """

    valid_result_regs: list[Register] = None
    """可以使用的结果寄存器
    """

    valid_func_arg_regs: list[Register] = None
    """可以使用的函数输入寄存器
    """

    def reset(self) -> None:
        """清除所有寄存器使用痕迹, 除了保留寄存器."""

        self.all_registers = AttrDict()

        local_regs = [f"R{i}" for i in range(10)]
        special_local_regs = ["R5_NEG", "R6_NEG", "ADD_S", "MUL_P"]
        shared_regs = [f"SR{i}" for i in range(18)]

        index = 0
        for name in local_regs + special_local_regs + shared_regs:
            if getattr(self, name, None) is None:
                setattr(self, name, Register().update(index=index, name=name))
            self.all_registers[index] = getattr(self, name)
            index += 1

        # 结果寄存器可以为 R0-4
        self.valid_result_regs = []
        for i in [4, 3, 2, 1, 0]:
            self.valid_result_regs += [self.all_registers[i]]

        # 函数输入寄存器可以为 R2-4, R7-9
        self.valid_func_arg_regs = []
        for i in [9, 8, 7, 4, 3, 2]:
            self.valid_func_arg_regs += [self.all_registers[i]]

    def __post_init__(self) -> None:
        """`dataclass` 构造函数, `__init__` 运行之后执行"""
        self.reset()


class RegisterCollection(RegisterCollectionBase):
    """寄存器集合, 用于 SMT 编译器."""

    _v: Register = None
    """函数参数寄存器 V.
    """

    _v_neg: Register = None
    """函数参数寄存器 V 的负值.
    """

    def get_reg_by_name(self, name: str) -> Register:
        """根据寄存器名字找到寄存器.

        Args:
            name (str): 寄存器名字.
        """
        for reg in self.all_registers.values():
            if reg.name == name:
                return reg
        raise ValueError(f"找不到寄存器 {name}")

    @property
    def V(self) -> Register:
        """函数参数寄存器 V.

        Returns:
            Register: 函数参数寄存器 V.
        """
        if self._v is None:
            self.V = self.R5
        return self._v

    @V.setter
    def V(self, value: Register) -> None:
        """设置函数参数寄存器 V.

        Args:
            value (Register): 函数参数寄存器 V.
        """
        if value not in [self.R5, self.R6]:
            raise RuntimeError(f"V 只能是 R5 或者 R6 而不是 {value}")

        if value == self.R6:
            self._v = self.R6.update(alias="V", used_by={-1})
            self._v_neg = self.R6_NEG.update(alias="V_NEG", used_by={-1})
            self.R5.release()
            self.R5_NEG.release()
        else:
            self._v = self.R5.update(alias="V", used_by={-1})
            self._v_neg = self.R5_NEG.update(alias="V_NEG", used_by={-1})
            self.R6.release()
            self.R6_NEG.release()

    @property
    def V_NEG(self) -> Register:
        """函数参数寄存器 V 的负数.

        Returns:
            Register: 函数参数寄存器 V 的负数.
        """
        if self._v_neg is None:
            self.V = self.R5
        return self._v_neg

    @V_NEG.setter
    def V_NEG(self, value: Register) -> None:
        """设置函数参数寄存器 V 的负数.

        Args:
            value (Register): 函数参数寄存器 V 的负数.
        """
        if value not in [self.R5_NEG, self.R6_NEG]:
            raise RuntimeError(f"V 只能是 R5 或者 R6 而不是 {value}")

        if value == self.R6_NEG:
            self._v = self.R6.update(name="V", used_by={-2})
            self._v_neg = self.R6_NEG.update(name="V_NEG", used_by={-2})
            self.R5.release()
            self.R5_NEG.release()
        else:
            self._v = self.R5.update(name="V", used_by={-2})
            self._v_neg = self.R5_NEG.update(name="V_NEG", used_by={-2})
            self.R6.release()
            self.R6_NEG.release()

    @cached_property
    def FAKE_NA_REG(self) -> Register:
        """占位符寄存器. 存储常数 7. 不可以用作输入, 可以用作结果寄存器."""
        return Register(index=7, _name="FAKE_NA_REG")

    @cached_property
    def ONE_REG(self) -> Register:
        """常数 1"""
        return self.SR16.update(alias="ONE_REG", value=1, used_by={-2})

    @cached_property
    def ZERO_REG(self) -> Register:
        """常数 0"""
        return self.SR17.update(alias="ZERO_REG", value=0, used_by={-2})

    @property
    def pos_reg(self) -> tuple[Register, Register]:
        """取负数使用的正值寄存器.

        Returns:
            Register: 正值寄存器.
        """
        if self.V == self.R5:
            return self.R6
        return self.R5

    @property
    def neg_reg(self) -> tuple[Register, Register]:
        """取负数使用的负值寄存器.

        Returns:
            Register: 负值寄存器.
        """
        if self.V == self.R5:
            return self.R6_NEG
        return self.R5_NEG

    @cached_property
    def shared_regs(self) -> list[Register]:
        """返回所有共享寄存器 SR0 至 SR17"""
        if not self.all_registers:
            self.reset()
        return [r for r in self.all_registers.values() if r.name.startswith("SR")]

    @property
    def unused_shared_reg(self) -> Register:
        """返回一个还没有被使用的共享寄存器 (`SR0` 到 `SR17`).

        Returns:
            Register: 还没有被使用的共享寄存器.
        """
        for sr in self.shared_regs:
            if not sr.used_by:
                return sr
        raise RuntimeError("找不到没有使用过的共享寄存器.")

    @property
    def unused_dummy_reg(self) -> Register:
        """返回一个虚拟寄存器 (`DUMMY_*`).
        最后这些虚拟寄存器会被合并到 R0-6.

        Returns:
            Register: 还没有被使用的虚拟寄存器.
        """
        index = len(self.all_registers)
        result = Register(_name=f"DUMMY_{index}", index=index)
        self.all_registers[index] = result
        return result

    @property
    def unused_arg_reg(self) -> Register:
        """返回一个还没有被使用的函数参数寄存器 (R2-4, R7-9).

        Returns:
            Register: 还没有被使用的函数参数寄存器.
        """
        for reg in self.valid_func_arg_regs:
            if reg.used_by:
                continue
            if reg in self.valid_result_regs:
                self.valid_result_regs.remove(reg)
            if reg in self.valid_func_arg_regs:
                self.valid_func_arg_regs.remove(reg)
            return reg
        raise RuntimeError("找不到未被占用的函数参数寄存器")

    @property
    def dummy_regs(self) -> list[Register]:
        """返回所有被使用的虚拟寄存器.

        Returns:
            list[Register]: 所有被使用的虚拟寄存器.
                用第一次使用的 IR 语句排序.
        """
        result = []

        for reg in self.all_registers.values():
            if reg.index < 32:
                continue
            if not reg.used_by:  # 跳过没用的结果寄存器
                continue
            if reg.name.startswith("Unused"):  # 跳过标记过的结果寄存器
                continue
            result += [reg]
        result = sorted(result, key=lambda r: r.first)
        return result


@dataclass
class SMT:  # pylint: disable=too-many-instance-attributes
    """SMT 指令."""

    op: Operator = Operator.NOP
    """6-bit 操作码
    """

    a1: Register = None
    """5-bit 加法源操作数 A1
    """

    a2: Register = None
    """5-bit 加法源操作数 A2
    """

    m1: Register = None
    """5-bit 乘法源操作数 M1
    """

    m2: Register = None
    """5-bit 乘法源操作数 M2
    """

    s: Register = None
    """3-bit 加法目标操作数 S. Default to register with index 7 means not applicable.
    """

    p: Register = None
    """3-bit 乘法目标操作数 P. Default to register with index 7 means not applicable.
    """

    operator: str = "unknown"
    """记录加法 `add`,  乘法 `mul` 或者乘加 `add_mul`
    """

    @property
    def input_regs(self) -> list[Register]:
        """返回输入寄存器. `self.a1`, `self.a2`, `self.m1`, `self.m2`

        Returns:
            list[Register]: 所有输入寄存器.
        """
        return [self.a1, self.a2, self.m1, self.m2]

    def update_regs(
        self,
        old_reg: Register,
        new_reg: Register,
        reg_names: list[str] = None,
    ) -> bool:
        """更新寄存器.

        Args:
            old_reg (Register): 旧寄存器.
            new_reg (Register): 新寄存器.

        Returns:
            bool: 更新寄存器成功.
        """
        reg_names = reg_names or ["a1", "a2", "m1", "m2", "s", "p"]
        result = False
        for reg_name in reg_names:
            if old_reg != getattr(self, reg_name, None):
                continue
            setattr(self, reg_name, new_reg)
            result = True
        return result

    def update_operand(self, old_reg: Register, new_reg: Register) -> bool:
        """更新操作数寄存器.

        Args:
            old_reg (Register): 旧的寄存器.
            new_reg (Register): 新的寄存器.

        Returns:
            bool: 更新了操作数.
        """
        if self.op != Operator.CALCU:
            return False
        return self.update_regs(old_reg=old_reg, new_reg=new_reg, reg_names=["a1", "a2", "m1", "m2"])

    @property
    def value(self) -> int:
        """只读 SMT 值. 根据当前 SMT 计算.

        Returns:
            int: 当前 SMT 的值.
        """
        result = []
        result += [bin(self.op.value)[2:].rjust(6, "0")]

        if self.a1 is None:
            result += ["00000"]
        else:
            result += [bin(self.a1.index)[2:].rjust(5, "0")]

        if self.a2 is None:
            result += ["00000"]
        else:
            result += [bin(self.a2.index)[2:].rjust(5, "0")]

        if self.s is None:
            result += ["000"]
        else:
            result += [bin(self.s.index)[2:].rjust(3, "0")]

        if self.m1 is None:
            result += ["00000"]
        else:
            result += [bin(self.m1.index)[2:].rjust(5, "0")]

        if self.m2 is None:
            result += ["00000"]
        else:
            result += [bin(self.m2.index)[2:].rjust(5, "0")]

        if self.p is None:
            result += ["000"]
        else:
            result += [bin(self.p.index)[2:].rjust(3, "0")]

        return "_".join(result)

    @property
    def reg_result(self) -> Register | tuple[Register, Register]:
        """结果寄存器

        Returns:
            Register | tuple[Register, Register]: 结果寄存器.
                如果操作为加乘则返回加法结果和乘法结果两个寄存器.
        """
        if self.op != Operator.CALCU:
            raise ValueError(f"操作 {self} 没有结果寄存器")

        if self.operator == "add":
            return self.s

        if self.operator == "mul":
            return self.p

        if self.operator == "add_mul":
            return self.s, self.p

        raise ValueError(f"不能得到结果寄存器: 未知操作 {self.operator}")

    @reg_result.setter
    def reg_result(self, value: Register) -> Register:
        """设置结果寄存器.

        Raises:
            NotImplementedError: 暂不支持乘法和加法同时运算.

        Returns:
            Register: 结果寄存器
        """
        if self.op != Operator.CALCU:
            raise ValueError(f"{self.op} 没有结果寄存器")

        if self.operator == "add":
            self.s = value
            return

        if self.operator == "mul":
            self.p = value
            return

        if self.operator == "add_mul":
            if isinstance(value, tuple) and len(value) == 2:
                self.s = value[0]
                self.p = value[1]
                return
            raise ValueError(f"运算的结果不能为 {value}.")

        raise ValueError(f"不能设置结果寄存器: 未知操作 {self.operator}")

    def __str__(self) -> str:
        if self.op == Operator.CALCU:
            sum_product = []
            if (self.a1.alias, self.a2.alias) != ("ZERO_REG", "ZERO_REG"):
                sum_product += [f"{self.s.short} = {self.a1.short} + {self.a2.short}"]
            if (self.m1.alias, self.m2.alias) != ("ZERO_REG", "ZERO_REG"):
                sum_product += [f"{self.p.short} = {self.m1.short} * {self.m2.short}"]
            return f"{self.op.name}: " + (", ".join(sum_product))
        if self.op == Operator.V_SET:
            return f"V_SET: delta V = {self.a1.name}, V_reset = {self.a2.name}:{self.a2.value}, V = {self.s.name}"
        if self.op == Operator.SPIKE:
            return f"SPIKE: delta V = {self.a1.name}"
        if self.op in [Operator.NOP, Operator.SRAM_LOAD, Operator.SRAM_SAVE, Operator.END, Operator.SPIKE]:
            return self.op.name
        result = [str(self.op)]
        for key in ["a1", "a2", "s", "m1", "m2", "p"]:
            reg = getattr(self, key)
            result += [f"{key.upper()}:{reg.name}({reg.index})"]
        return ", ".join(result)

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class SMTFactory:
    """SMT 生成器."""

    regs: RegisterCollection = field(default_factory=RegisterCollection)
    """编译用到的寄存器
    """

    def get_reg(self, int_or_reg: Register | int) -> Register:
        """根据数值或寄存器返回寄存器对象.

        Args:
            int_or_reg (Register | int): 寄存器对象或寄存器数值. 支持 0 和 1.

        Returns:
            Register: 寄存器对象.
        """
        if isinstance(int_or_reg, Register):
            return int_or_reg

        if int_or_reg == 0:
            return self.regs.ZERO_REG

        if int_or_reg == 1:
            return self.regs.ONE_REG

        raise ValueError(f"不支持的 {int_or_reg = }")

    def add(self, a: Register | int, b: Register | int, c: Register | None = None) -> list[SMT]:
        """加法 c = a + b. MUL_P = 0 * 0.

        Args:
            a (Register | int): 被加数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            b (Register | int): 加数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            c (Register, optional): 结果寄存器.

        Returns:
            list[SMT]: SMT 语句
        """

        result = [
            SMT(
                op=Operator.CALCU,
                a1=self.get_reg(a),
                a2=self.get_reg(b),
                s=c or self.regs.ADD_S,
                m1=self.regs.ZERO_REG,
                m2=self.regs.ZERO_REG,
                p=self.regs.FAKE_NA_REG,
                operator="add",
            ),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
        ]
        return result

    def move(self, src: Register, dst: Register) -> list[SMT]:
        """通过加零来移动 `src` 寄存器的值到 `dst` 寄存器.

        Args:
            src (Register): 源寄存器.
            dst (Register): 目标寄存器.

        Returns:
            list[SMT]: SMT 语句
        """
        return self.add(src, 0, dst)

    def multiply(self, a: Register | int, b: Register | int, c: Register | None = None) -> list[SMT]:
        """乘法 ADD_S = 0 + 0, c = a * b,

        Args:
            a (Register | int): 被乘数数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            b (Register | int): 乘数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            c (Register, optional): 结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """

        result = [
            SMT(
                op=Operator.CALCU,
                a1=self.regs.ZERO_REG,
                a2=self.regs.ZERO_REG,
                s=self.regs.FAKE_NA_REG,
                m1=self.get_reg(a),
                m2=self.get_reg(b),
                p=c or self.regs.MUL_P,
                operator="mul",
            ),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
        ]
        return result

    # pylint: disable-next=too-many-arguments
    def add_multiply(
        self,
        a1: Register,
        a2: Register,
        m1: Register,
        m2: Register,
        s: Register | None = None,
        p: Register | None = None,
    ) -> list[SMT]:
        """加法和乘法同时运算.

        - `s = a1 + a2`
        - `p = m1 * m2`

        Args:
            a1 (Register | int): 被加数数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            a2 (Register | int): 加数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            s (Register, optional): 和结果寄存器
            m1 (Register | int): 被乘数数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            m2 (Register | int): 乘数寄存器, 0: SMTReg.ZERO_REG, 1: SMTReg.ONE_REG
            p (Register, optional): 积结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """

        result = [
            SMT(
                op=Operator.CALCU,
                a1=a1,
                a2=a2,
                s=s or self.regs.ADD_S,
                m1=m1,
                m2=m2,
                p=p or self.regs.MUL_P,
                operator="add_mul",
            ),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
            SMT(op=Operator.NOP),
        ]
        return result

    def sram_load(self) -> list[SMT]:
        """读取 SRAM.

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(op=Operator.SRAM_LOAD)]

    def v_set(self, delta_v: Register, v_reset: Register) -> list[SMT]:
        """更新 V

        Args:
            delta_v (Register): V_thresh - V 结果寄存器
            v_reset (Register): v_reset 寄存器

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(op=Operator.V_SET, a1=delta_v, a2=v_reset, s=self.regs.V)]

    def spike(self, delta_v: Register) -> list[SMT]:
        """Spike

        Args:
            delta_v (Register): V_thresh - V 结果寄存器

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(Operator.SPIKE, a1=delta_v)]

    def sram_save(self) -> list[SMT]:
        """存储 SRAM.

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(Operator.SRAM_SAVE)]

    def end(self) -> list[SMT]:
        """结束.

        Returns:
            list[SMT]: SMT 语句
        """
        return [SMT(Operator.END)]
