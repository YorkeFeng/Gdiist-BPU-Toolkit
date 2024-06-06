"""stablehlo to SMT
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from functools import cached_property, reduce
from typing import Callable, Dict, List, Set, Tuple, Union

import jax
from addict import Dict as AttrDict
from brainpy._src.integrators import JointEq
from ortools.sat.python import cp_model
from strenum import StrEnum

from .smt import SMT, Number, Operator, Register, SMTFactory
from .stablehlo_parser import stablehlo_parser


class IROperandType(StrEnum):
    """操作数类型枚举Number = int | float

    - `arg_index`: 函数输入
    - `reg_index`: 结果寄存器
    - `constant`: 常数
    """

    arg_index = "arg_index"
    """函数输入索引
    """

    reg_index = "arg_index"
    """结果寄存器索引
    """

    constant = "constant"
    """常数数值
    """


@dataclass
class IROperand:
    """IR 中的操作数

    - `type`: `IROperandType`
    - `value`: `Number`
    """

    type: IROperandType
    """操作数类型
    """

    value: Number
    """操作数数值, 根据 `OperandType` 代表不同含义.

    - type == OperandType.arg_index, 数值为函数输入索引
    - type == OperandType.reg_index, 数值为结果寄存器索引
    - type == OperandType.constant, 数值为常数数值
    """


class IRCmd(StrEnum):
    """支持的 IR 命令类型"""

    negate = "negate"
    """取负值

    - 常数, 正值不被直接使用, 负值被使用: 存储负值到共享寄存器
    - 常数, 正值和负值都被使用: 存储正值和负值到两个共享寄存器
    - 运算结果, 结果不被直接使用, 负值被使用: 存储运算结果到 `smt.R6` 或 `smt.R5`,
        存储 `smt.R6_NEG`+ `ZERO_REG` 到共享寄存器
    - 运算结果, 结果和负值都被使用: 存储运算结果到 `smt.R6` 或 `smt.R5`,
        存储 (`smt.R6` 或 `smt.R5`) + `ZERO_REG` 到共享寄存器
        存储 (`smt.R6_NEG` 或 `smt.R5_NEG`) + `ZERO_REG` 到另一个共享寄存器
    - 优先使用 `smt.R6`, 如果被 V 占用则使用 `smt.R5`.
    """

    constant = "constant"
    """常数

    直接保存到共享寄存器
    """

    add = "add"
    """加法

    直接输出和到共享寄存器
    """

    multiply = "multiply"
    """乘法

    直接输出乘积到共享寄存器
    """

    divide = "divide"
    """除法

    - 除数为常数, 只被除法 `IRCmd.divide`: 存储除数的倒数到共享寄存器
        - 优化方向: 类型转换 `IRCmd.convert`, 取负值 `IRCmd.negate` 不占用寄存器
    - 除数为常数, 被其他语句使用: 存储常数和除数的倒数到共享寄存器
        - 优化方向: 类型转换 `IRCmd.convert`, 取负值 `IRCmd.negate` 不占用寄存器
    - 除数为运算结果: 不支持

    之后运行乘法并输出乘积到共享寄存器
    """

    power = "power"
    """指数, 暂时只支持非负整数常数.

    直接输出乘积到共享寄存器.
    """

    convert = "convert"
    """类型转换

    直接输出到共享寄存器
    """


@dataclass
class IRStatement:
    """IR 语句"""

    reg_index: int
    """运算结果寄存器索引
    """

    cmd: IRCmd
    """命令类型
    """

    operands: List[Union[Register, int]]
    """操作数

    - 寄存器对象: 函数输入或者常数
    - 结果寄存器索引: 运算结果
    """


class ParsedIR(AttrDict):
    """解析过的 IR.

    - `module_head`: 函数信息.
    - `func_head`: 函数头.
    - `func_body`: 函数体.
    """

    func_group: FuncGroup
    """函数组对象
    """

    @classmethod
    def load_one_func(cls, func: Callable) -> ParsedIR:
        """从函数得到的解析过的 stablehlo IR.

        Args:
            func (Callable): 函数

        Returns:
            ParsedIR: 解析过的 stablehlo IR.
        """
        func_sig = inspect.signature(func)
        arg_count = len(str(func_sig).split(","))
        lower_arg = list(range(arg_count))
        func_jit = jax.jit(func)
        func_ir = str(func_jit.lower(
            *lower_arg).compiler_ir(dialect="stablehlo"))
        result = ParsedIR(stablehlo_parser.parse_string(func_ir).as_dict())
        del result.func_body.return_statement["dtype"]
        return result

    @classmethod
    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def load(cls, func: Dict[str, Callable]) -> ParsedIR:
        """从函数得到的解析过的 stablehlo IR.

        Args:
            func (dict[str, Callable]): 函数返回变量名称 -> 函数.

        Returns:
            ParsedIR: 解析过的 stablehlo IR.
        """

        result = ParsedIR()
        result.func_group = Func.load(func)
        module_head = None

        func_head = AttrDict()
        func_head.name = "@main"
        func_head.arg_defs = []
        func_head.return_def = AttrDict()

        func_body = AttrDict()
        func_body.statements = []

        known_args = set()
        reg_index_base = 0

        for one_func in result.func_group:
            one_func: Func
            parsed_ir = ParsedIR.load_one_func(one_func.body)
            module_head = module_head or parsed_ir.module_head

            old_args = parsed_ir.func_arg_names

            for one_arg_def in parsed_ir.func_head.arg_defs:
                add_arg = ""
                for one_info in one_arg_def.info:
                    if one_info.name == ["jax", "arg_info"]:
                        add_arg = one_info.value
                        break
                if not add_arg:
                    continue
                if add_arg in known_args:
                    continue
                known_args.add(add_arg)
                arg_def = AttrDict()
                arg_def.index = len(func_head.arg_defs)
                arg_def.dtype = one_arg_def.dtype
                arg_def.info = one_arg_def.info
                func_head.arg_defs.append(arg_def)

            for stmt in parsed_ir.func_body.statements:
                stmt.reg_index = len(func_body.statements)
                for opr in stmt.operands:
                    if opr.type == "reg_index":
                        opr.value += reg_index_base
                    elif opr.type == "arg_index":
                        opr.value = old_args[opr.value]
                func_body.statements.append(stmt)

            reg_index_base = len(func_body.statements)

            if func_body.return_statement:
                func_body.return_statement.return_dtypes += parsed_ir.func_body.return_statement.return_dtypes
                parsed_ir.func_body.return_statement.operands[0].value = func_body.statements[-1].reg_index
                func_body.return_statement.operands += parsed_ir.func_body.return_statement.operands
            else:
                func_body.return_statement = parsed_ir.func_body.return_statement

            func_body.return_statement.operands[-1].name = one_func.returns

        module_head.at_identifier = "@jit_" + \
            ("_".join(result.func_group.returns))
        result.module_head = module_head
        result.func_head = func_head
        result.func_body = func_body
        if "dtype" in result.func_body.return_statement:
            del result.func_body.return_statement["dtype"]

        return result

    @property
    def func_arg_names(self) -> Dict[int, str]:
        """输入函数的参数名称.

        Returns:
            AttrDict[int, str]: 函数参数, e.g. `{0: "V", 1: "I"}`
        """
        result = AttrDict()
        for arg in self.func_head.arg_defs:
            for info in arg.info:
                if "arg_info" in info.name:
                    name = info.value
                    break
            else:
                raise RuntimeError(f"函数参数没有名字: {arg}")
            result[arg.index] = name
        return result


@dataclass
class Func:
    """一个函数, 包括:

    - `args`: 函数参数变量名字
    - `body`: 函数对象
    - `results`: 函数结果变量名字
    """

    args: list[str]
    """函数参数
    """

    body: Callable
    """函数算式
    """

    returns: str
    """函数结果变量名字
    """

    @classmethod
    def load(cls, func: Dict[str, Callable]) -> FuncGroup:
        """读取函数或函数组为 `Func` 对象.

        Args:
            func (dict[str, Callable]): 函数返回值名称 -> 函数或函数组 (`JointEq`).
                如果 func 的值是函数组 (`JointEq`) 则忽略返回值名称.

        Returns:
            FuncGroup: 函数组对象
        """
        result = FuncGroup()
        for return_name, one_func in func.items():
            if isinstance(one_func, JointEq):  # 读取函数组信息
                # 虽然可以通过函数名称推导返回值名称但是怕 brainpy 内部结构改变
                for i, result_vars in enumerate(one_func.vars_in_eqs):
                    result += [Func(args=one_func.args_in_eqs[i],
                                    body=one_func.eqs[i], returns=result_vars[0])]
            else:
                parsed_ir = ParsedIR.load_one_func(one_func)
                result += [Func(args=list(parsed_ir.func_arg_names.values()),
                                body=one_func, returns=return_name)]
        return result


class FuncGroup(list):
    """函数组, 对象属性:

    - `args`: 所有函数的参数变量名字
    - `results`: 所有函数的结果变量名字
    """

    @property
    def args(self) -> List[str]:
        """所有函数的参数.

        Returns:
            list[str]: 所有函数的参数.
        """
        result = []
        for one_func in self:
            if not isinstance(one_func, Func):
                raise RuntimeError(
                    f"{one_func} is NOT Func but {type(one_func)}")
            for one_arg in one_func.args:
                if one_arg not in result:
                    result += [one_arg]
        return result

    @property
    def returns(self) -> List[str]:
        """所有函数的结果变量名.

        Returns:
            list[str]: 所有函数的结果变量名.
        """
        result = []
        for one_func in self:
            if not isinstance(one_func, Func):
                raise RuntimeError(
                    f"{one_func} is NOT Func but {type(one_func)}")

            if one_func.returns not in result:
                result += [one_func.returns]
        return result


@dataclass
class SMTCompilerBase:  # pylint: disable=too-many-instance-attributes
    """SMT 编译器"""

    # region: 构造函数参数 `func`, `i_func`, `preload_constants`, `fixed_constants`
    # 必要参数
    func: Dict[str, Callable]
    """需要编译的函数返回值名称和函数体.
    """

    is_i_func: bool = False
    """函数是 I 运算.
    """

    i_reg_name: str = ""
    """I 参数的寄存器名字, e.g. R3
    """

    reg_map: Dict[str, str] = field(default_factory=AttrDict)
    """函数输入寄存器, e.g. {"g1": "R3"}
    """

    update_method: Dict[str, str] = field(default_factory=AttrDict)
    """结果更新方法, 默认 acc 累加.
    """

    used_arg_names: Dict[str, str] = field(default_factory=AttrDict)
    """使用的输入变量名称到寄存器名称的映射.
    """

    used_shared_regs: Set[Register] = field(default_factory=set)
    """使用的共享寄存器.
    """

    smt_factory: SMTFactory = None
    """SMT 生成工厂.
    """

    # 可选参数
    preload_constants: Set[Register] = field(default_factory=set)
    """常数寄存器, 需要预先加载到 `property.bin`.
    `ZERO_REG` 和 `ONE_REG` 会在构造函数里添加到 `preload_constants`.
    """

    # 可选参数
    fixed_constants: Set[Register] = field(default_factory=set)
    """必须放在固定位置的常数. 默认

    - SR0 = V_reset 值为 0
    - SR1 = T_refrac 值为 0
    """
    # endregion: 构造函数参数 `func`, `preload_constants`, `fixed_constants`

    # region: 私有成员
    _reg_results: Dict[int, Register] = field(default_factory=AttrDict)
    """每一条 IR 语句运算的结果
    """

    _smt_results: Dict[int, list[SMT]] = field(default_factory=AttrDict)
    """编译结果, SMT 语句
    """

    _v_thresh: Register = None
    """必须存在的 V_thresh 寄存器.
    """

    _v_reset: Register = None
    """必须存在的 V_reset 寄存器.
    """
    # endregion: 私有成员

    smt_info_str: str = ""
    """未优化的 SMT 语句.
    """

    def __post_init__(self) -> None:
        """构造函数"""

        if self.smt_factory is None:
            self.smt_factory = SMTFactory()

        self.fixed_constants = {
            self.smt_factory.regs.SR0.update(used_by={-2}, alias="V_reset"),
            self.smt_factory.regs.SR1.update(used_by={-2}, alias="T_refrac"),
        }

        self.preload_constants |= self.fixed_constants
        self.preload_constants.add(
            self.smt_factory.regs.ZERO_REG.update(used_by={-2}))
        self.preload_constants.add(
            self.smt_factory.regs.ONE_REG.update(used_by={-2}))

    @cached_property
    def log(self) -> logging.Logger:
        """日志记录器 默认等级 `logging.INFO`

        Returns:
            logging.Logger: 日志记录器
        """
        result = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        return result

    @cached_property
    def parsed_ir(self) -> ParsedIR:
        """从函数得到的解析过的 stablehlo IR.

        如果输入是函数组`JointEq`, 则合并所有函数体.

        Returns:
            ParsedIR: 解析过的 stablehlo IR.
        """
        result = ParsedIR.load(self.func)
        for i, return_name in enumerate(self.func):
            result.func_body.return_statement.operands[i].name = return_name
        return result

    @cached_property
    def func_args(self) -> Dict[int, Register]:
        """输入函数的参数.

        Returns:
            AttrDict[int, Register]: 函数参数, e.g. `{0: V, 1: I}`,
        """
        result: Dict[int, Register] = AttrDict()
        arg_names = []
        for i, (name, reg_name) in enumerate(self.used_arg_names.items()):
            reg = self.smt_factory.regs.get_reg_by_name(reg_name)
            reg.update(alias=name, used_by={-1}, as_arg=name)
            result[i] = reg
            arg_names.append(name)
            if name not in self.reg_map:
                continue

            if self.reg_map[name] != reg.name:
                err_msg = f"寄存器冲突: {name} 需要放在 reg_map 中的 {self.reg_map[name]} "
                err_msg += f"而不是 used_reg_names 中的 {reg.name}."
                raise RuntimeError(err_msg)

        for i, name in self.parsed_ir.func_arg_names.items():
            if name in arg_names:
                continue
            if (not self.is_i_func) and name.startswith("I") and self.i_reg_name:
                # I 计算过, 使用之前的寄存器
                reg = self.smt_factory.regs.get_reg_by_name(
                    name=self.i_reg_name)
            elif name == "V":
                reg = self.smt_factory.regs.V
            else:
                if name in self.reg_map:
                    reg = self.smt_factory.regs.get_reg_by_name(
                        self.reg_map[name])
                    if reg.used_by:
                        raise RuntimeError(f"{name} 不能放在已被占用的 {reg.name}.")
                else:
                    reg = self.smt_factory.regs.unused_arg_reg

            if name.startswith("I"):
                name = "I"
            reg.update(alias=name, used_by={-1}, as_arg=name)
            result[len(result)] = reg
        for reg in result.values():
            if reg in self.smt_factory.regs.valid_result_regs:
                self.smt_factory.regs.valid_result_regs.remove(reg)
            if reg in self.smt_factory.regs.valid_func_arg_regs:
                self.smt_factory.regs.valid_func_arg_regs.remove(reg)
        return result

    @cached_property
    def return_names(self) -> Dict[int, str]:
        """返回所有输出变量的名字, e.g. {8: "V"} 表示第 8 条语句的结果为 V 输出.
        注意: 运算阶段不要更新.

        Returns:
            dict[int, str]: 所有输出语句编号和对应变量名称.
        """
        result: Dict[int, str] = AttrDict()
        for opr in self.parsed_ir.func_body.return_statement.operands:
            result[opr.value] = opr.name
        return result

    def get_result_reg(self, stmt_id: int) -> Register:
        """返回没占用的共享寄存器作为结果寄存器并记录在 `self._reg_results`.
        `result.used_by = {stmt_id}`.

        Returns:
            Register: 结果寄存器
        """
        if self._reg_results[stmt_id]:
            raise RuntimeError(f"结果寄存器 {stmt_id} 已经被占用.")

        result = self.smt_factory.regs.unused_dummy_reg

        if stmt_id in self.return_names:
            result.as_return = self.return_names[stmt_id]
            return_index = next(i for i, v in enumerate(
                self.return_names.values()) if v == result.as_return)
            result.used_by.add(
                return_index + len(self.parsed_ir.func_body.statements))

        result.used_by.add(stmt_id)
        self._reg_results[stmt_id] = result
        return result

    def update_result_reg(self, stmt_id: int, reg: Register) -> None:
        """使用 `reg` 更新 `self._reg_results` 以及 `ir_stmts` 操作数寄存器.

        Args:
            stmt_id (int): IR 语句索引.
            new_reg (Register): 新的寄存器.
        """
        old_reg: Register = self._reg_results[stmt_id]
        replaced = False
        for ir_stmt in self.ir_stmts:
            if old_reg not in ir_stmt.operands:
                continue
            for i, r in enumerate(ir_stmt.operands):
                if old_reg == r:
                    ir_stmt.operands[i] = reg
                    replaced = True
        if replaced:
            reg.used_by = old_reg.used_by | reg.used_by
            old_reg.release()
        self._reg_results[stmt_id] = reg

    def add_constant_reg(self, value: Number) -> Register:
        """添加常数寄存器.

        - 如果常数已存在, 返回常数寄存器.
        - 如果常数不存在, 返回新构造的常数寄存器.

        Returns:
            Register: 常数寄存器
        """
        for reg in self.preload_constants:
            if reg.value == value:
                return reg

        result = self.smt_factory.regs.unused_shared_reg
        result.value = value
        self.preload_constants.add(result)
        return result

    @cached_property
    def ir_stmts(self) -> List[IRStatement]:
        """IR 指令列表. 操作数为:

        - 寄存器对象: 函数输入或者常数
        - 结果寄存器索引: 运算结果

        Returns:
            list[IRStatement]: IR 指令列表.
        """
        result: List[IRStatement] = []
        for stmt_id, stmt in enumerate(self.parsed_ir.func_body.statements):
            ir_stmt = IRStatement(reg_index=stmt_id, cmd=stmt.cmd, operands=[])
            self.get_result_reg(stmt_id=stmt_id)  # 初始化结果寄存器占位符
            for opr in stmt.operands:
                if opr.type == "arg_index":  # 函数输入
                    if opr.value.startswith("I"):
                        opr.value = "I"
                    reg = next(r for r in self.func_args.values()
                               if r.as_arg == opr.value)
                elif opr.type == "constant":  # 常数
                    reg = self.add_constant_reg(opr.value)
                elif opr.type == "reg_index":  # 之前的计算结果
                    reg = self._reg_results[opr.value]  # IR 语句索引 == 结果寄存器索引
                else:
                    raise NotImplementedError(
                        f"暂不支持 {opr.type = }, {opr.value = }")
                # 先记录寄存器使用, 如果不需要占用再删除. 比如 `constant`, `convert`.
                if reg not in [self.smt_factory.regs.ONE_REG, self.smt_factory.regs.ZERO_REG]:
                    reg.used_by.add(stmt_id)
                ir_stmt.operands += [reg]
            result += [ir_stmt]
        return result


class SMTCompiler(SMTCompilerBase):
    """SMT 编译器"""

    _compiled: bool = False
    """已经编译过了
    """

    def get_stmt_id_by_reg(self, reg: Register) -> int:
        """根据结果寄存器对象, 返回 IR 语句的索引值.

        Args:
            reg (Register): 结果寄存器.

        Returns:
            int: IR 语句的索引值.
        """
        for result, result_reg in self._reg_results.items():
            if result_reg == reg:
                return result
        for arg_reg in self.func_args.values():
            if arg_reg == reg:
                return -1
        raise RuntimeError(f"{reg} 不是结果寄存器也不是函数输入寄存器.")

    def get_used_by_others(self, stmt_id: int, reg: Register) -> Set[int]:
        """返回使用寄存器的其他语句.

        Args:
            stmt_id (int): 当前语句索引.
            reg (Register): 寄存器对象.

        Returns:
            set[int]: 使用寄存器的其他语句索引
        """
        return reg.used_by - {-1, -2, stmt_id}

    def get_negated(self, stmt_id: int, opr: Register) -> Register:
        """得到取负值的运算结果. 具体方法参见 `IRCmd.negate`.

        Args:
            stmt_id (int): 当前 IR 语句索引.
            opr (Register): 操作数寄存器.

        Returns:
            Register: 负数结果寄存器. `used_by` 已更新
        """
        # 找到使用正值的语句, 不包括当前语句.
        used_by_others = self.get_used_by_others(stmt_id=stmt_id, reg=opr)

        # 根据 V 使用的寄存器得到正值寄存器和负值寄存器
        r6 = self.smt_factory.regs.pos_reg
        r6_neg = self.smt_factory.regs.neg_reg

        if opr in self.preload_constants:  # 常数的负数
            sr_x_neg = self.add_constant_reg(-(opr.value))
            sr_x_neg.used_by |= {stmt_id}
            sr_x_neg.alias = f"-{opr.short}"

            if (not used_by_others) and (opr not in [self.smt_factory.regs.SR0, self.smt_factory.regs.SR1]):
                remember_reg = self.smt_factory.regs.unused_dummy_reg
                remember_reg.update(
                    name=f"Unused {opr.name}",
                    value=opr.value,
                    alias=opr.alias,
                    used_by=opr.used_by,
                )
                sr_x_neg.alias = f"{-opr.value}"
                opr.release()
                self.preload_constants.discard(opr)

            return sr_x_neg

        if opr.as_arg:  # 输入的负数
            # r6 = opr
            self._smt_results[stmt_id] += self.smt_factory.move(
                src=opr, dst=r6)
            return r6_neg

        # 运算结果的负数
        # 找到之前的运算语句
        # ir_stmt_id = -1 则为函数输入
        ir_stmt_id = self.get_stmt_id_by_reg(opr)

        # 1. 将之前的运算结果保存在正值寄存器, e.g. `R6`
        if ir_stmt_id == -1:  # 函数输入的负数
            # 在当前 SMT 语句块将其复制 (R6 = opr + 0) 到 `R6` 取负值.
            self._smt_results[stmt_id] += self.smt_factory.move(
                src=opr, dst=r6)
        else:  # 运算语句的结果
            # 如果有运算, 则直接改变运算结果寄存器为 `R6`.
            # 先不改变结果寄存器.
            for ir_stmt in reversed(self._smt_results[ir_stmt_id]):
                if ir_stmt.op == Operator.CALCU:
                    ir_stmt.reg_result = r6
                    break
            else:
                # 如果没有运算比如 convert
                # 则在运算语句所在的语句块将其复制到 `R6` 取负值.
                self._smt_results[ir_stmt_id] += self.smt_factory.move(
                    src=opr, dst=r6)

        sr_x_neg = self.smt_factory.regs.unused_dummy_reg  # 负值结果
        sr_x_neg.used_by.add(stmt_id)
        if ir_stmt_id > -1:
            sr_x_neg.used_by.add(ir_stmt_id)

        # 没有其他语句使用正值
        if (not used_by_others) or (used_by_others == set([ir_stmt_id])):
            # 2. 添加 SMT 语句: SR_X_NEG = R6_NEG + ZERO_REG
            # 不能直接使用 r6_neg 因为可能会被其他语句占用
            self._smt_results[ir_stmt_id] += self.smt_factory.move(
                src=r6_neg, dst=sr_x_neg)

            # 3. 更新运算结果寄存器为 SR_X_NEG
            # 没有其他运算使用这个结果
            self._reg_results[ir_stmt_id] = self.smt_factory.regs.FAKE_NA_REG

            # 5. 释放 opr
            opr.release()
            return sr_x_neg

        # 有其他语句使用正值
        # 2. 添加 SMT 语句: SR_X_NEG = R6_NEG + ZERO_REG; SR_X = R6 * ONE_REG
        sr_x = self.smt_factory.regs.unused_dummy_reg  # 正值, 用来代替 opr
        sr_x.used_by = opr.used_by  # 使用 SR_X 代替 opr

        # SR_X_NEG = R6_NEG + ZERO_REG; SR_X = R6 * ONE_REG
        self._smt_results[ir_stmt_id] += self.smt_factory.add_multiply(
            a1=r6_neg,
            a2=0,
            m1=r6,
            m2=1,
            s=sr_x_neg,
            p=sr_x,
        )

        # 3. 更新运算结果寄存器为 SR_X
        self._reg_results[ir_stmt_id] = sr_x

        # 4. 使用 SR_X 代替 opr
        for used_stmt_id in opr.used_by:  # 每一条 IR 语句
            if used_stmt_id < 0:  # 跳过函数输入和保留常数
                continue
            # IR 语句对应的一组 SMT 语句
            for smt_cmd in self._smt_results[used_stmt_id]:
                smt_cmd: SMT
                smt_cmd.update_operand(old_reg=opr, new_reg=sr_x)  # 更新寄存器

        return sr_x_neg

    def cmd_constant(self, stmt_id: int) -> Register:
        """常数命令.

        操作数和结果是同一个寄存器对象.
        不记录占用, 因为肯定会被其它语句占用.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        # 常数寄存器输入已经在 `SMTCompilerBase.ir_stmts` 中
        # 保存在 `SMTCompilerBase.preload_constants`.
        # 常数命令的操作数和结果是同一个寄存器对象
        # 释放占位符
        reg = self.ir_stmts[stmt_id].operands[0]
        self.update_result_reg(stmt_id=stmt_id, reg=reg)

        # 不记录占用, 因为肯定会被其它语句占用.
        reg.used_by.discard(stmt_id)

        return reg

    def cmd_convert(self, stmt_id: int) -> Register:
        """类型转换命令.

        操作数和结果是同一个寄存器对象.
        不记录占用, 因为肯定会被其它语句占用.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        return self.cmd_constant(stmt_id=stmt_id)

    def cmd_add(self, stmt_id: int) -> Register:
        """加法.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        opr = self.ir_stmts[stmt_id].operands
        result = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.add(
            a=opr[0], b=opr[1], c=result)
        return result

    def cmd_multiply(self, stmt_id: int) -> Register:
        """乘法.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        opr = self.ir_stmts[stmt_id].operands

        if self.smt_factory.regs.ZERO_REG in opr:
            self._reg_results[stmt_id] = self.smt_factory.regs.ZERO_REG
            return self.smt_factory.regs.ZERO_REG

        if self.smt_factory.regs.ONE_REG in opr:
            value_regs = opr.copy()
            value_regs.remove(self.smt_factory.regs.ONE_REG)
            if value_regs:
                self.update_result_reg(stmt_id=stmt_id, reg=value_regs[0])
            else:
                self.update_result_reg(
                    stmt_id=stmt_id, reg=self.smt_factory.regs.ONE_REG)
            return self._reg_results[stmt_id]

        result: Register = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.multiply(
            opr[0], opr[1], result)
        return result

    def cmd_power(self, stmt_id: int) -> Register:
        """指数.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """
        opr = self.ir_stmts[stmt_id].operands

        if opr[1] not in self.preload_constants:
            raise NotImplementedError(f"暂不支持非常数次幂 {opr[1]}")

        if opr[1].value != int(opr[1].value):
            raise NotImplementedError(f"暂不支持非整数次幂 {opr[1].value}")

        opr[1].value = int(opr[1].value)

        if opr[1].value < 0:
            raise NotImplementedError(f"暂不支持负数次幂 {opr[1].value}")

        if opr[1].value == 0:
            self.update_result_reg(
                stmt_id=stmt_id, reg=self.smt_factory.regs.ONE_REG)
            return self.smt_factory.regs.ONE_REG

        if opr[1].value == 1:
            return self.cmd_convert(stmt_id)

        last_result = self.smt_factory.regs.unused_dummy_reg
        result = last_result
        last_result.used_by.add(stmt_id)

        self._smt_results[stmt_id] += self.smt_factory.multiply(
            a=opr[0], b=opr[0], c=result)
        for _ in range(opr[1].value - 2):
            result = self.smt_factory.regs.unused_dummy_reg
            result.used_by.add(stmt_id)
            self._smt_results[stmt_id] += self.smt_factory.multiply(
                a=last_result, b=opr[0], c=result)
            last_result = result

        self.update_result_reg(stmt_id=stmt_id, reg=result)
        return result

    def cmd_negate(self, stmt_id: int) -> Register:
        """取负值运算. 具体方法参见 `IRCmd.negate`.

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """

        opr = self.ir_stmts[stmt_id].operands

        if opr[0] == self.smt_factory.regs.V:  # 如果 V 取负值, 返回 V_NEG
            result = self.smt_factory.regs.V_NEG
        else:  # 其他情况返回生成的负值
            result = self.get_negated(stmt_id=stmt_id, opr=opr[0])

        self.update_result_reg(stmt_id=stmt_id, reg=result)
        return result

    def cmd_divide(self, stmt_id: int) -> Register:
        """除法

        - 除数为常数, 只被除法 IRCmd.divide 使用: 存储除数的倒数到共享寄存器
        - 除数为常数, 被除法 IRCmd.divide 和其他运算使用: 存储常数和除数的倒数到共享寄存器
        - 除数为运算结果: 不支持

        之后运行乘法并输出乘积到共享寄存器

        优化方向:

        - 如果除数只被类型转换 IRCmd.convert 使用, 那么不用存储
        - 如果除数只被取负值 IRCmd.negate 使用, 那么不用存储

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """

        opr = self.ir_stmts[stmt_id].operands

        if opr[1] not in self.preload_constants:
            raise NotImplementedError("暂只支持除以常数")

        # 如果除数没有被其他语句使用, 不保存除数
        used_by_others = self.get_used_by_others(stmt_id=stmt_id, reg=opr[1])

        # 记录除数的倒数
        divider = self.add_constant_reg(1 / (opr[1].value))
        divider.used_by.add(stmt_id)
        divider.alias = f"1/{opr[1].short}"

        if not used_by_others:  # 如果除数没有被其他语句使用, 不保存除数
            if opr[1] not in (self.smt_factory.regs.SR0, self.smt_factory.regs.SR1):
                divider.alias = f"1/{opr[1].value}"
                opr[1].release()
                self.preload_constants.discard(opr[1])

        # 运行乘法代替除法
        result = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.multiply(
            a=opr[0], b=divider, c=result)
        return result

    def cmd_subtract(self, stmt_id: int) -> Register:
        """减法

        - 减数为常数, 只被减法 `IRCmd.subtract` 使用: 存储减数的负数到共享寄存器.
        - 减数为常数, 被减法 `IRCmd.subtract` 和其他运算使用: 存储常数和减数的负数到共享寄存器.
        - 减数为运算结果: 取负数.

        之后运行加法法并输出和到共享寄存器

        Args:
            stmt_id (int): IR 语句索引.

        Returns:
            Register: 结果寄存器.
        """

        opr = self.ir_stmts[stmt_id].operands

        # 负值
        sr_x_neg = self.get_negated(stmt_id=stmt_id, opr=opr[1])

        # 运行加法代替减法
        result = self._reg_results[stmt_id]
        self._smt_results[stmt_id] += self.smt_factory.add(
            a=opr[0], b=sr_x_neg, c=result)
        return

    def compile(self) -> None:
        """得到 SMT 语句和预加载常数."""

        if self._compiled:
            return

        self.used_arg_names = {
            r.alias: r.name for r in self.func_args.values()}

        for stmt_id, stmt in enumerate(self.ir_stmts):
            cmd_method = getattr(self, f"cmd_{stmt.cmd}", None)
            if cmd_method is None:
                raise NotImplementedError(f"暂不支持指令 {stmt.cmd}")
            cmd_method(stmt_id)

        # 保存结果的第一句语句编号
        smt_stmt_id = len(self.parsed_ir.func_body.statements)
        i_reg = None
        for i, (ir_id, arg_name) in enumerate(self.return_names.items()):
            # 在输入的输出, e.g. V, 累加
            if arg_name in self.parsed_ir.func_arg_names.values():
                arg_reg = next(r for r in self.func_args.values()
                               if r.alias == arg_name)
                update_method = self.update_method.get(arg_name, "acc")
                if update_method == "acc":
                    self._reg_results[smt_stmt_id] = arg_reg
                    arg_reg.used_by.add(smt_stmt_id)
                    # 累加, e.g. V = dV + V
                    self._smt_results[smt_stmt_id] += self.smt_factory.add(
                        a=self._reg_results[ir_id],
                        b=arg_reg,
                        c=arg_reg,
                    )
                    smt_stmt_id += 1
                elif update_method == "update":
                    self._reg_results[ir_id].release()
                    self._reg_results[ir_id] = arg_reg
                    for stmt in self._smt_results[ir_id]:
                        stmt: SMT
                        if stmt.op != Operator.CALCU:
                            continue
                        stmt.reg_result = arg_reg

            else:
                # 不在输入的输出, e.g. I, 替换 dummy 寄存器为 R2-4
                for reg_id in [2, 3, 4]:
                    reg = self.smt_factory.regs.all_registers[reg_id]
                    if not reg.used_by:
                        arg_reg = reg
                        break
                else:
                    raise RuntimeError("R2-4 寄存器不够.")
                arg_reg.update(alias=arg_name, used_by={ir_id})
                for smt in self._smt_results[ir_id]:
                    smt: SMT
                    if smt.op != Operator.CALCU:
                        continue
                    if smt.s == self._reg_results[ir_id]:
                        smt.s = arg_reg
                    if smt.p == self._reg_results[ir_id]:
                        smt.p = arg_reg
            if self.is_i_func and arg_name == "I":
                i_reg = arg_reg

        if self.is_i_func:
            self._compiled = True
            self.i_reg_name = i_reg.name
            return

        stmt_offset = len(self._reg_results)

        # Delta V = R0 = V_thresh - V
        self._smt_results[stmt_offset + 0] += self.smt_factory.add(
            a=self._v_thresh,
            b=self.smt_factory.regs.V_NEG,
            c=self.smt_factory.regs.R0,
        )

        # 根据 Delta V 设置 V = V_reset
        self._smt_results[stmt_offset + 1] += self.smt_factory.v_set(
            self.smt_factory.regs.R0, self._v_reset)

        # 根据 Delta V 发出激励
        self._smt_results[stmt_offset +
                          1] += self.smt_factory.spike(self.smt_factory.regs.R0)

        self._smt_results[stmt_offset + 1] += self.smt_factory.sram_save()
        self._smt_results[stmt_offset + 1] += self.smt_factory.end()

        self._compiled = True
        return

    def optimize(self) -> Dict[Register, Register]:
        """优化寄存器使用.

        Returns:
            dict[Register, Register]: 合并映射.
        """
        reg_map = {}
        self.try_use_result_reg()
        reg_map = self.merge_regs()
        for old_reg, new_reg in reg_map.items():
            old_reg: Register
            old_reg.replace_by(new_reg)
        return reg_map

    def merge_regs(self) -> Dict[Register, Register]:
        """合并虚拟寄存器到 `R0` 和 `R1`

        Returns:
            dict[Register, Register]: 合并映射.
        """

        if not (regs_to_merge := self.smt_factory.regs.dummy_regs):
            return {}

        result: Dict[Register, Register] = AttrDict()
        # 可以使用的结果寄存器
        result_regs = len(self.smt_factory.regs.valid_result_regs)

        # print()
        # print("合并之前:")
        # for reg in regs_to_merge:
        #     print(reg)
        # print()

        # region: 约束编程模型
        model = cp_model.CpModel()
        max_value = max(max(reg.used_by) for reg in regs_to_merge)

        reg_ids = []  # 放到第几个空余寄存器
        usages = []
        for i, reg in enumerate(regs_to_merge):
            reg_ids += [model.NewIntVar(0, result_regs - 1,
                                        f"real_reg_index_for_reg_{i}")]
            usages += [
                model.NewFixedSizeIntervalVar(
                    start=(reg_ids[-1] * max_value) + reg.first + 1,
                    size=reg.last - reg.first,
                    name=f"usage_{i}",
                )
            ]

        model.AddNoOverlap(usages)
        model.Minimize(sum(reg_ids))
        # endregion: 约束编程模型

        # region: 约束编程求解
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        for status_str in ["unknown", "model_invalid", "feasible", "infeasible", "optimal"]:
            if status == getattr(cp_model, status_str.upper()):
                status_str = status_str.upper()
                break

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            err_msg = [f"求解失败: {status_str}."]
            err_msg += [f"{self.smt_info_str}"]
            err_msg += ["函数输入:"]
            for line in self.parsed_ir["func_body"]["statements"]:
                err_msg += [str(line)]
            raise RuntimeError("\n".join(err_msg))
        # endregion: 约束编程求解

        for old_reg_id, real_reg_id in enumerate(reg_ids):
            reg_id = len(self.smt_factory.regs.valid_result_regs) - \
                solver.Value(real_reg_id) - 1
            result[regs_to_merge[old_reg_id]
                   ] = self.smt_factory.regs.valid_result_regs[reg_id]
            result[regs_to_merge[old_reg_id]
                   ].used_by |= regs_to_merge[old_reg_id].used_by
        return result

    def try_use_result_reg(self) -> None:
        """合并虚拟寄存器到 `ADD_S` 或者 `MUL_P`.

        只有马上使用到的且以后不会再使用的结果会使用 `ADD_S` 或者 `MUL_P`.

        更新 `self._reg_results`.
        """

        not_nop_stmts: List[Tuple[int, SMT]] = []
        for ir_stmt_id, smt_stmts in self._smt_results.items():
            smt_stmts: List[SMT]
            for smt_stmt in smt_stmts:
                if smt_stmt.op == Operator.NOP:
                    continue
                not_nop_stmts += [(ir_stmt_id, smt_stmt)]

        for i, (ir_stmt_id, smt_stmt) in enumerate(not_nop_stmts):
            if i == len(not_nop_stmts) - 1:
                break

            ir_stmt_id: int
            smt_stmt: SMT

            if smt_stmt.operator == "add":
                result_reg = smt_stmt.s
                new_reg = self.smt_factory.regs.ADD_S
            elif smt_stmt.operator == "mul":
                result_reg = smt_stmt.p
                new_reg = self.smt_factory.regs.MUL_P
            else:
                continue

            if not result_reg.name.startswith("DUMMY_"):
                continue

            if len(result_reg.used_by) > 2:
                continue

            next_stmt = not_nop_stmts[i + 1][1]

            if result_reg not in next_stmt.input_regs:
                continue

            multiple_usage = False
            for _, stmt in not_nop_stmts[i + 2:]:
                if result_reg in stmt.input_regs:
                    multiple_usage = True
                    break

            if multiple_usage > 1:
                continue

            # print(f"使用 {new_reg} 代替 {result_reg}.")
            next_stmt.update_regs(old_reg=result_reg, new_reg=new_reg)

            result_reg.release()

            if smt_stmt.operator == "add":
                smt_stmt.s = self.smt_factory.regs.ADD_S
            else:
                smt_stmt.p = self.smt_factory.regs.MUL_P

    _final_smt_result: List[SMT] = []
    """SMT 语句
    """

    def get_smt_result(self) -> List[SMT]:
        """返回得到 SMT 语句"""

        if self._final_smt_result:
            return self._final_smt_result

        self.compile()

        smt_info = []
        smt_info += [""]
        smt_info += ["未合并的虚拟寄存器"]
        for reg in self.smt_factory.regs.all_registers.values():
            if reg.index < 32:
                continue
            if reg.used_by:
                smt_info += [str(reg)]
        smt_info += [""]
        smt_info += ["未优化的 SMT 语句"]
        for stmts in self._smt_results.values():
            for stmt in stmts:
                if stmt.op == Operator.NOP:
                    continue
                smt_info += [str(stmt)]

        self.smt_info_str = "\n".join(smt_info)
        reg_map = self.optimize()
        result: List[SMT] = reduce(
            lambda a, b: a + b, self._smt_results.values())
        for smt_cmd in result:
            smt_cmd: SMT
            if smt_cmd.op != Operator.CALCU:
                continue

            for arg_name in ["a1", "a2", "s", "m1", "m2", "p"]:
                if (arg_reg := getattr(smt_cmd, arg_name)) not in reg_map:
                    continue
                setattr(smt_cmd, arg_name, reg_map[arg_reg])

            # 使用 3'b111 作为计算结果表示输出到 ADD_S 或 MUL_P
            if smt_cmd.reg_result in [self.smt_factory.regs.ADD_S, self.smt_factory.regs.MUL_P]:
                smt_cmd.reg_result = self.smt_factory.regs.FAKE_NA_REG
        self.used_shared_regs = {
            sr for sr in self.smt_factory.regs.shared_regs if sr.used_by}
        self._final_smt_result = result
        return self._final_smt_result

    @classmethod
    def compile_all(
        cls,
        func: Dict[str, Callable],
        preload_constants: Dict[str, Number] = None,
        i_func: Callable = None,
        reg_map: Dict[str, str] = None,
        update_method: Dict[str, str] = None,
    ) -> Tuple[SMTCompiler, SMTCompiler, list[SMT]]:
        """编译 I 函数和其他函数.

        Args:
            func (dict[str, Callable]): 函数, 比如 {"V": xxx, "u": xxx}
            preload_constants (dict[str, Number]): 预先载入常数.
            i_func (Callable, Optional): I 函数.
            reg_map (Dict[str, str], Optional): 函数输入寄存器的存放寄存器, e.g. {"g1": "R3"}

        Returns:
            tuple[SMTCompiler, SMTCompiler, list[SMT]]: I 函数编译器,
                其他编译器, SMT 语句.
        """
        reg_map = reg_map or {}
        update_method = update_method or {}
        constants = AttrDict(preload_constants)
        v_compiler = cls(func=func, is_i_func=False,
                         reg_map=reg_map, update_method=update_method)
        compilers = [v_compiler]
        if i_func:
            i_compiler = cls(func={"I": i_func}, is_i_func=True,
                             reg_map=reg_map, update_method=update_method)
            compilers = [i_compiler] + compilers
        else:
            i_compiler = None

        smt_result = []
        i_reg_name = ""
        used_arg_names = AttrDict()
        used_shared_regs = set()
        used_regs = set()
        for compiler in compilers:
            regs = compiler.smt_factory.regs
            pc = compiler.preload_constants
            for reg_name in used_regs:
                if reg_name in ["R0", "R1"]:
                    continue
                reg = regs.get_reg_by_name(reg_name)
                reg.used_by.add(-3)  # 被 I 使用

            # region: 预先载入常数
            pc.add(regs.SR0.update(alias="v_reset",
                   value=constants['V_reset'], used_by={-2}))
            pc.add(regs.SR1.update(alias="t_refrac",
                   value=constants['T_refrac'], used_by={-2}))
            pc.add(regs.SR2.update(alias="v_thresh",
                   value=constants['V_thresh'], used_by={-2}))
            pc.add(regs.SR3.update(alias="v_rest",
                   value=constants['V_rest'], used_by={-2}))
            compiler._v_thresh = regs.SR2  # pylint: disable=protected-access
            compiler._v_reset = regs.SR0  # pylint: disable=protected-access
            # endregion: 预先载入常数

            if not compiler.is_i_func:
                compiler.i_reg_name = i_reg_name
                compiler.used_arg_names = used_arg_names
                compiler.used_shared_regs = used_shared_regs
                for old_reg in compiler.used_shared_regs:
                    new_reg = compiler.smt_factory.regs.all_registers[old_reg.index]
                    new_reg.used_by.add(-2)
                    new_reg.update(value=old_reg.value, alias=old_reg.alias)
                    pc.add(new_reg)

            compiler.compile()
            smt_result += compiler.get_smt_result()

            for reg in compiler.smt_factory.regs.all_registers.values():
                if reg.used_by:
                    used_regs.add(reg.name)

            if compiler.is_i_func:
                i_reg_name = compiler.i_reg_name
                used_shared_regs = compiler.used_shared_regs
                used_arg_names = compiler.used_arg_names

        smt_result = v_compiler.smt_factory.sram_load() + smt_result
        return (i_compiler, v_compiler, smt_result)
