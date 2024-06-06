"""这个模块包含了用于处理SMT 96位指令的常用函数和类
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from .smt_96_base import REG_INDEX, IBinary


@dataclass
class RegisterBase:
    """寄存器基类"""

    index: int = -1
    """寄存器索引, 默认值 -1, 表示上一个寄存器 index + 1
    """

    name: str = None
    """汇编语言中的寄存器名称, e.g. R0, 默认值 None.
    """

    used_by: set[int] = field(default_factory=set)
    """使用此寄存器的 IR 语句, int: 语句的索引值

    - -1 表示函数参数, e.g. `I`
    - -2 表示保留寄存器, e.g. `NONE_REG`, `ZERO_REG`
    - -3 表示被之前的编译过程使用过的寄存器
    """

    value: int = 0
    """当前寄存器的值, 默认值 0
    """

    alias: str = None
    """SMT 语句中的寄存器别名, e.g. `V_reset`, 默认值 None
    """

    as_arg: str = None
    """寄存器作为函数输入的名字, e.g. `I`, 默认值 None
    """

    as_return: str = None
    """寄存器作为函数输出的名字, e.g. `I`, 默认值 None
    """

    @property
    def short(self) -> str:
        """寄存器的短名称: 别名或者名称

        Returns:
            str: 寄存器的短名称
        """
        return self.as_arg or self.alias or self.name

    def update(self, **kwargs: Any) -> RegisterBase:
        """改变成员的值

        Args:
            kwargs (Any): 改变寄存器成员的值

        Returns:
            RegisterBase: 修改过的当前寄存器
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"{self.__class__.__name__} 没有 {k} 成员")
            setattr(self, k, v)
        return self

    def __hash__(self) -> int:
        """取得哈希码 这样寄存器就可以放到 set 里边了

        **相同 name 和 index 的寄存器会有相同的哈希码**

        Returns:
            int: 哈希码
        """
        return hash(f"{self.name}, {self.index}")

    def __str__(self) -> str:
        """返回字符转

        Example:
            >>> reg = RegisterBase(index=1, used_by={1, 2}, value=3, alias="V_reset", as_arg="I", as_return="I")
            >>> reg.__str__()
            '<R1(V_reset) = 3, used by: [1, 2], func_arg: I, func_return: I>'

        Returns:
            str: 字符串
        """
        result = self.name or f"R{self.index}"
        if self.alias:
            result += f"({self.alias})"
        result += f" = {self.value}, used by: {sorted(self.used_by)}"
        if self.as_arg:
            result += f", func_arg: {self.as_arg}"
        if self.as_return:
            result += f", func_return: {self.as_return}"
        result = f"<{result}>"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def release(self) -> None:
        """释放寄存器, 重置寄存器的值和使用者

        >>> reg = RegisterBase(index=1, name="R1", used_by={1, 2}, value=3, alias="V_reset", as_arg="I", as_return="I")
        >>> reg.index
        1
        >>> reg.used_by
        {1, 2}
        >>> reg.__dict__
        {'index': 1, 'name': 'R1', 'used_by': {1, 2}, 'value': 3, 'alias': 'V_reset', 'as_arg': 'I', 'as_return': 'I'}
        >>> reg.release()
        >>> reg.index
        1
        >>> reg.used_by
        set()
        >>> reg.__dict__
        {'index': 1, 'name': 'R1', 'used_by': set(), 'value': 0, 'alias': None, 'as_arg': None, 'as_return': None}
        """
        self.update(**RegisterBase(index=self.index, name=self.name).__dict__)

    def replace_by(self, reg: RegisterBase) -> RegisterBase:
        """用 `reg` 的信息替换当前寄存器的信息, 但是保留 `used_by` 属性

        >>> r1 = RegisterBase(index=1, used_by={1, 2}, name="R1")
        >>> r2 = RegisterBase(index=2, used_by={3, 4}, name="R2")
        >>> r1.replace_by(r2).__str__()
        '<R2 = 0, used by: [1, 2]>'

        Args:
            reg (RegisterBase): 另一个寄存器

        Returns:
            RegisterBase: 修改过的当前寄存器
        """
        used_by = self.used_by
        self.update(**reg.__dict__)
        self.used_by = used_by
        return self

    @property
    def used_by_list(self) -> list[int]:
        """返回 `RegisterBase.used_by` 列表

        Returns:
            list[int]: `RegisterBase.used_by` 列表
        """
        return sorted(list(self.used_by))

    @property
    def first(self) -> int:
        """找到第一次使用的语句, i.e. `RegisterBase.used_by` 最小值

        Raises:
            ValueError: 寄存器没有使用者

        Returns:
            int: `RegisterBase.used_by` 最小值
        """
        if not self.used_by:
            raise ValueError("找不到第一次使用的语句, 寄存器没有使用者.")
        return min(self.used_by)

    @property
    def last(self) -> int:
        """找到最后一次使用的语句, i.e. `RegisterBase.used_by` 最大值

        Raises:
            ValueError: 寄存器没有使用者

        Returns:
            int: `RegisterBase.used_by` 最大值
        """
        if not self.used_by:
            raise ValueError("找不到最后一次使用的语句, 寄存器没有使用者")
        return max(self.used_by)


class Register96(RegisterBase):
    """SMT 96-bit 寄存器."""

    @property
    def operand(self) -> str:
        """5 位二进制操作数

        Returns:
            str: 5 位二进制操作数
        """
        return IBinary.dec2bin(self.index, 5)


class RegisterCollection:
    """寄存器集合"""

    regs: dict[int, Register96]
    """所有数据寄存器. `SMT96Factory` 对象可以通过索引或寄存器名称得到寄存器对象
    """

    _dummy_index: int
    """虚拟寄存器的索引"""

    def __init__(self) -> None:
        """构造函数

        >>> regs = RegisterCollection()
        >>> regs.ZERO_REG.name
        'ZERO_REG'
        >>> regs.ZERO_REG.index
        16
        >>> regs.NONE_REG.name
        'NONE_REG'
        >>> regs.NONE_REG.index
        31
        >>> regs.NONE_REG.used_by
        {-2}
        >>> regs.CTRL_PULSE
        <CTRL_PULSE = 0, used by: [-2]>
        """
        self.regs = {}
        self._dummy_index = 1024

        for i in range(32):
            self.regs[i] = Register96(index=i, name=f"R{i}")

        for reg_index in REG_INDEX:
            self.regs[reg_index.value].update(name=reg_index.name, used_by={-2})

    def __getitem__(self, key: Union[int, str]) -> Register96:
        """通过索引或名称得到寄存器

        >>> regs = RegisterCollection()
        >>> regs[31]
        <NONE_REG = 0, used by: [-2]>
        >>> regs[16]
        <ZERO_REG = 0, used by: [-2]>
        >>> regs["NONE_REG"]
        <NONE_REG = 0, used by: [-2]>
        >>> regs["ZERO_REG"]
        <ZERO_REG = 0, used by: [-2]>
        >>> regs.NONE_REG.used_by
        {-2}

        Args:
            key (Union[int, str]): 寄存器索引(index)或名称(name)或别名(alias)

        Returns:
            Register96: 寄存器对象
        """
        if isinstance(key, int):
            return self.regs[key]

        if isinstance(key, str):
            for reg in self.regs.values():
                if key in [reg.name, reg.alias]:
                    return reg

        raise KeyError(f"找不到寄存器 {key = }.")

    def __getattr__(self, name: str) -> Register96:
        """通过属性名得到寄存器

        >>> regs = RegisterCollection()
        >>> regs.ZERO_REG.index
        16
        >>> regs.NONE_REG.index
        31

        Args:
            name (str): 寄存器名称

        Returns:
            Register96: 寄存器对象
        """
        return self[name]

    def get_unused_reg(self) -> Register96:
        """返回一个未被占用的寄存器

        Returns:
            Register96: 还没有被占用的寄存器
        """
        for i in range(16):
            if not (reg := self[i]).used_by:
                return reg
        raise RuntimeError("找不到未被占用的寄存器")

    def get_dummy_reg(self) -> Register96:
        """返回一个虚拟寄存器 (`DUMMY_*`)
        最后这些虚拟寄存器会被合并到未使用的真实寄存器

        Returns:
            Register96: 虚拟寄存器
        """
        index = self._dummy_index
        result = Register96(name=f"DUMMY_{index}", index=index)
        self.regs[index] = result
        self._dummy_index += 1

        return result

    def release(self, reg: Register96) -> None:
        """释放寄存器

        Args:
            reg (Register96): 要释放的寄存器
        """
        if reg.index > 32:
            del self.regs[reg.index]
        else:
            reg.release()

    @property
    def dummy_regs(self) -> list[Register96]:
        """返回当前所有虚拟寄存器 (`DUMMY_*`)

        Returns:
            Register96: 虚拟寄存器
        """
        result: list[Register96] = []
        for reg in self.regs.values():
            if reg.index >= 32 and reg.used_by:
                result.append(reg)
        return result

    @property
    def unused_regs(self) -> list[Register96]:
        """返回当前所有未被占用的寄存器

        Returns:
            Register96: 闲置寄存器
        """
        result: list[Register96] = []
        for reg in self.regs.values():
            if reg.index < 16 and not reg.used_by:
                result.append(reg)
        return result

    @property
    def used_regs(self) -> list[Register96]:
        """返回当前所有被占用的寄存器

        Returns:
            Register96: 被占用的寄存器
        """
        result: list[Register96] = []
        for reg in self.regs.values():
            if reg.index < 16 and reg.used_by:
                result.append(reg)
        return result


RegOrConstant = Union[Register96, float]
"""类型: 寄存器或者数值
"""

__all__ = [
    "RegisterBase",
    "Register96",
    "RegisterCollection",
    "RegOrConstant",
]
