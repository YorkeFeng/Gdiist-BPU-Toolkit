# pylint: disable=line-too-long
"""这个模块包含了用于处理 SMT 96 位指令的常用函数和类
"""

from __future__ import annotations

import math
import stat
from dataclasses import dataclass, field
from functools import cached_property, partial, reduce
from typing import Callable, Union

from addict import Dict as AttrDict
from loguru import logger
from ortools.sat.python import cp_model

from ..backend.smt_96_stmt.fp_op import FP_OP
from ..common.smt_96_base import CTRL_LEVEL, CTRL_PULSE, IBinary

from ..backend.smt_96_factory import SMT96Factory
from ..backend.smt_96_stmt import NOP, SMT96
from ..common.smt_96_reg import Register96, RegisterCollection
from ..frontend.stablehlo_parser import StableHLOProgram, StableHLOStatement


@dataclass
class SMT96CompilerBase:
    """SMT 96-bit 编译器基础成员

    >>> import brainpy as bp
    >>> funcs = {"V": bp.neurons.LIF(256).derivative}
    >>> SMT96CompilerBase(funcs=funcs).stablehlo_program.func_body.return_statement.operands[0].name
    'V'
    >>> funcs = {"HindmarshRose": bp.neurons.HindmarshRose(256).derivative}
    >>> SMT96CompilerBase(funcs=funcs).stablehlo_program.func_body.return_statement.operands[0].name
    'HindmarshRose'
    >>> compiler = SMT96CompilerBase(funcs = {"V": bp.neurons.LIF(256).derivative})
    >>> compiler.regs[3]
    <R3 = 0, used by: []>
    >>> compiler.stablehlo_statements[4]
    [StableHLOStatement(reg_index=4, cmd='convert', operands=[<R1(t) = 0, used by: [-1, 4], func_arg: t>])]
    """

    funcs: dict[str, Callable]
    """需要编译的函数返回值名称和函数体
    """

    regs: RegisterCollection = None
    """寄存器集合, 如果为 None 则在构造函数中赋值为 `self.smt_factory.regs`"""

    smt_factory: SMT96Factory = field(init=False)
    """SMT 语句生成器"""

    stmt_results: dict[int, Register96] = field(default_factory=AttrDict)
    """每一条 StableHLO 语句运算的结果. key 为语句的索引值, value 为结果寄存器
    """

    constants: dict[str, Union[int, float]] = field(default_factory=dict)
    """常数, e.g. V_th
    """

    def __post_init__(self) -> None:
        """构造函数

        >>> import brainpy as bp
        >>> from ..common.smt_96_base import REG_INDEX, CTRL_LEVEL
        >>> compiler = SMT96CompilerBase(funcs = {"V": bp.neurons.LIF(256).derivative})
        >>> compiler.regs["ZERO_REG"]
        <ZERO_REG = 0, used by: [-2]>
        """
        self.regs = self.regs or RegisterCollection()
        self.smt_factory = SMT96Factory(regs=self.regs)

    @cached_property
    def stablehlo_program(self) -> StableHLOProgram:
        """StableHLO 程序, 如果输入是函数组`JointEq`, 则合并所有函数体

        Returns:
            StableHLOProgram: StableHLO 程序
        """
        result = StableHLOProgram.load(self.funcs)
        for i, return_name in enumerate(self.funcs):
            # 添加 name 属性, 用于记录结果寄存器的名称, e.g. name = "V"
            result.func_body.return_statement.operands[i].name = return_name
        return result

    @cached_property
    def func_args(self) -> dict[int, Register96]:
        """函数参数编号到寄存器的映射, key 为参数的编号, value 为参数的寄存器,
        e.g. `{0: V, 1: I}`

        Returns:
            AttrDict[int, Register]: 函数参数编号到寄存器的映射, e.g. `{0: V, 1: I}`
        """
        result: dict[int, Register96] = {}

        for arg_name in self.stablehlo_program.func_arg_names.values():
            if arg_name.startswith("I"):  # 所有以 I 开头的参数都强制改名为 I
                arg_name = "I"
            try:  # 之前编译过程使用过的参数寄存器需要被重用
                reg = next(r for r in self.regs.used_regs if arg_name in [r.alias, r.as_arg])
            except StopIteration:  # 分配参数寄存器, 之前使用的寄存器不可以被使用
                reg = self.regs.get_unused_reg().update(used_by={-1})
            reg.update(alias=arg_name, as_arg=arg_name)
            result[len(result)] = reg

        # 上一次使用的寄存器可以再次被中间变量使用
        for reg in self.regs.used_regs:
            if reg.used_by == {-3}:
                self.regs.release(reg)

        return result

    @cached_property
    def return_names(self) -> dict[int, str]:
        """返回所有输出变量的名字,
        key 为输出 StableHLO 语句编号, value 为输出变量名称,
        e.g. {8: "V"} 表示第 8 条语句的结果为函数结果 V

        Returns:
            dict[int, str]: 所有输出语句编号和对应变量名称
        """
        result: dict[int, str] = {}
        for opr in self.stablehlo_program.func_body.return_statement.operands:
            # opr.name 属性是 `StableHLOProgram.load` 方法添加的
            result[opr.value] = opr.name
        return result

    @cached_property
    def stablehlo_statements(self) -> dict[int, list[StableHLOStatement]]:
        """返回更新过操作数的 StableHLO 指令列表,

        `self.stmt_results`

        - 操作数从字符串转换为寄存器或常数

        Returns:
            dict[int, list[StableHLOStatement]]: IR 指令列表
        """
        result: dict[int, list[StableHLOStatement]] = {}
        for stmt_id, stmt in enumerate(self.stablehlo_program.statement_list):
            self.get_result_reg(stmt_id=stmt_id)  # 确保每一条语句都有一个结果寄存器
            operands = []
            for opr in stmt.operands:
                opr_type, opr_value = opr.split(":")
                opr_value = opr_value.strip()
                if opr_type == "arg_index":
                    opr_value = int(opr_value)
                    reg = self.func_args[opr_value]
                    reg.used_by.add(stmt_id)
                    operands.append(reg)
                elif opr_type == "constant":
                    opr_value = float(opr_value)
                    operands.append(opr_value)
                elif opr_type == "reg_index":
                    opr_value = int(opr_value)
                    reg = self.stmt_results[opr_value]
                    reg.used_by.add(stmt_id)
                    operands.append(reg)
                else:
                    raise NotImplementedError(f"暂不支持 {opr_type = }, {opr_value = }")
                stmt.operands = operands
            result[stmt_id] = [stmt]
        return result

    def get_result_reg(self, stmt_id: int) -> Register96:
        """返回 `stmt_id` 语句的结果寄存器,
        如果未分配则为其分配一个虚拟寄存器, 并记录在 `self.stmt_results`

        Returns:
            Register96: 结果寄存器
        """
        if stmt_id in self.stmt_results:
            return self.stmt_results[stmt_id]

        result = self.regs.get_dummy_reg()
        result.used_by.add(stmt_id)

        if stmt_id in self.return_names:
            result.as_return = self.return_names[stmt_id]
            result.alias = self.return_names[stmt_id]
            return_index = next(i for i, v in enumerate(self.return_names.values()) if v == result.as_return)
            # 所有语句执行完之后, 再逐一保存函数结果, 所以也需要被使用一次
            result.used_by.add(return_index + len(self.stablehlo_program.func_body.statements))

        self.stmt_results[stmt_id] = result
        return result

    def replace_result_reg(self, old_reg: Register96, new_item: Union[Register96, float] = None) -> None:
        """使用 `new_item` 更新结果寄存器和语句中的操作数与结果寄存器

        - 更新 `self.stmt_results`
        - 更新 `self.stablehlo_statements`

        Args:
            old_reg (Register96): 旧寄存器
            new_item (Union[Register96, float]): 新的寄存器或者常数
        """
        if not isinstance(new_item, (Register96, float)):
            raise TypeError(f"不支持 {type(new_item) = }")

        # 更新操作数寄存器
        replaced = False
        statements = reduce(lambda a, b: a + b, list(self.stablehlo_statements.values()))
        for sh_stmt in statements:
            if sh_stmt.reg_index == old_reg.index and isinstance(new_item, Register96):
                sh_stmt.reg_index = new_item.index
                replaced = True
            if old_reg not in sh_stmt.operands:
                continue
            for opr_index, opr in enumerate(sh_stmt.operands):
                if old_reg == opr:
                    sh_stmt.operands[opr_index] = new_item
                    replaced = True

        # 更新结果寄存器
        for stmt_id, reg in self.stmt_results.items():
            if old_reg != reg:
                continue
            self.stmt_results[stmt_id] = new_item
            replaced = True

        if not replaced:
            raise ValueError(f"没有找到需要替换的寄存器 {old_reg = } {new_item = }")

        if isinstance(new_item, Register96):
            new_item.used_by = old_reg.used_by | new_item.used_by

        self.regs.release(old_reg)


@dataclass
class SMT96Compiler(SMT96CompilerBase):
    """SMT 96-bit 编译器

    1. 从函数得到的解析过的 stableHLO, 如果输入是函数组`JointEq`, 则合并所有函数体
    2. 对于每一条 `self.stablehlo_statements[stmt_id]`
        - 修改结果寄存器 `self.stmt_results[stmt_id]`
        - 修改 `cmd`, e.g. `divide` 变成 `multiply`
        - 修改 `operands`, 保存寄存器对象和数值
        - 修改 `reg_index`, 保存结果寄存器对象

    Example:
        >>> import brainpy as bp
        >>> funcs = {"V": bp.neurons.LIF(256).derivative}
        >>> compiler = SMT96Compiler(funcs=funcs)
        >>> compiler.compile()
        [R2 = 0.00 - R0, NOP, NOP, NOP, NOP, R2 = 0.00 + R2, NOP, NOP, NOP, NOP, R3 = 1.00 * R1, NOP, NOP, NOP, NOP, R2 = R2 + R3, NOP, NOP, NOP, NOP, R2 = 0.10 * R2, NOP, NOP, NOP, NOP]

    """

    _compiled: bool = False
    """是否已经编译"""

    _stablehlo_statements_updated: bool = False
    """是否已经更新了 `self.stablehlo_statements`"""

    smt96_results: list[SMT96] = field(default_factory=list)
    """编译结果, SMT 语句"""

    def check_type(self, operands: list, types: list[type], reporter: str) -> None:
        """检查类型是否支持

        Args:
            operands (list): 操作数
            types (list[type]): 支持的类型
            reporter (str): 报告者
        """

        for i, (opr, opr_type) in enumerate(zip(operands, types)):
            if isinstance(opr, opr_type):
                continue
            raise TypeError(f"{reporter}: 不支持非 {opr_type}, {type(operands[i]) = }")

    def update_stablehlo_statements(self) -> None:
        """更新 `self.stablehlo_statements`

        >>> import brainpy as bp
        >>> funcs = {"V": bp.neurons.LIF(256).derivative}
        >>> compiler = SMT96Compiler(funcs=funcs)
        >>> compiler.update_stablehlo_statements()
        >>> compiler._stablehlo_statements_updated
        True
        >>> compiler.stablehlo_statements[4]
        []

        """
        if self._stablehlo_statements_updated:
            return

        self._stablehlo_statements_updated = True

        for stmt_id, stmts in self.stablehlo_statements.items():
            stmt = stmts[0]
            old_reg = self.stmt_results[stmt_id]
            stmt.reg_index = old_reg.index
            operands = stmt.operands
            if stmt.cmd == "negate":
                self.check_type(operands, [Register96], stmt.cmd)
                stmt.cmd = "subtract"
                stmt.operands = [0, operands[0]]
            elif stmt.cmd == "convert":
                self.check_type(operands, [(Register96, float)], stmt.cmd)
                self.stablehlo_statements[stmt_id] = []  # 删除 convert 语句
                self.replace_result_reg(old_reg=old_reg, new_item=operands[0])
            elif stmt.cmd == "constant":
                self.check_type(operands, [float], stmt.cmd)
                self.replace_result_reg(old_reg=old_reg, new_item=operands[0])
                self.stablehlo_statements[stmt_id] = []  # 删除 constant 语句
            elif stmt.cmd in ["add", "multiply", "subtract"]:
                # review: 优化方向 x1, +0, -0
                self.check_type(operands, [(Register96, float), (Register96, float)], stmt.cmd)
            elif stmt.cmd == "divide":
                self.check_type(operands, [(Register96, float), float], stmt.cmd)
                if isinstance(operands[0], float):  # 纯常数运算
                    self.replace_result_reg(old_reg=old_reg, new_item=operands[0] / operands[1])
                    self.stablehlo_statements[stmt_id] = []  # 删除 divide 语句
                else:
                    stmt.cmd = "multiply"
                    stmt.operands = [operands[0], 1.0 / operands[1]]
            elif stmt.cmd == "power":
                self.update_power(stmt_id, stmt)
            else:
                raise NotImplementedError(f"暂不支持指令 {stmt.cmd}")

        return

    def update_power(self, stmt_id: int, stmt: StableHLOStatement) -> None:
        """更新 power 语句"""

        operands = stmt.operands
        self.check_type(operands, [(Register96, float), float], stmt.cmd)
        self.stablehlo_statements[stmt_id] = []

        if isinstance(operands[0], float):
            self.replace_result_reg(old_reg=self.stmt_results[stmt_id], new_item=operands[0] ** operands[1])
            return

        if operands[1] == 1:
            self.replace_result_reg(old_reg=self.stmt_results[stmt_id], new_item=operands[0])
            return

        if int(operands[1]) != operands[1]:
            raise ValueError(f"power: 暂不支持浮点数指数 {operands[1] = }")

        for _ in range(int(operands[1]) - 2):
            self.stablehlo_statements[stmt_id].append(
                StableHLOStatement(
                    reg_index=operands[0].index,
                    cmd="multiply",
                    operands=[operands[0], operands[0]],
                )
            )
        self.stablehlo_statements[stmt_id].append(
            StableHLOStatement(
                reg_index=self.stmt_results[stmt_id].index,
                cmd="multiply",
                operands=[operands[0], operands[0]],
            )
        )

    def get_return_reg(self, return_name: str) -> Register96:
        """返回函数结果寄存器

        Args:
            return_name (str): 函数结果名称

        Returns:
            Register96: 函数结果寄存器
        """
        for stmt_id, name in self.return_names.items():
            if return_name == name:
                return self.stmt_results[stmt_id]
        raise ValueError(f"没有找到函数结果寄存器 {return_name = }")

    # pylint: disable-next=too-many-locals
    def merge_regs(self) -> None:
        """合并虚拟寄存器

        寄存器分配问题可以看作是一个约束编程问题,
        整个解空间可以看作为一个 2 维空间,
        一个维度是空闲寄存器的编号
        另一个维度是该寄存器的使用情况
        通过 ortools.sat.python.cp_model.CpModel 求解器求解

        - 使得寄存器的编号尽可能小
        - 不发生冲突
        """

        if not (regs_to_merge := self.regs.dummy_regs):
            return

        regs_to_merge: list[Register96]
        unused_regs = self.regs.unused_regs

        # region: 约束编程模型 model
        model = cp_model.CpModel()
        max_value = max(reg.last for reg in regs_to_merge)

        reg_ids = []  # 空闲寄存器编号
        usages = []  # 寄存器使用情况

        for i, reg in enumerate(regs_to_merge):
            reg_ids.append(
                model.NewIntVar(
                    0,
                    len(unused_regs) - 1,
                    f"real_reg_index_for_reg_{i}",
                )
            )
            usages.append(
                model.NewFixedSizeIntervalVar(
                    start=(reg_ids[-1] * max_value) + reg.first + 1,
                    size=reg.last - reg.first,
                    name=f"usage_{i}",
                )
            )

        model.Minimize(sum(reg_ids))  # 使得寄存器的编号尽可能小
        model.AddNoOverlap(usages)  # 不发生冲突
        # endregion: 约束编程模型 model

        # region: 约束编程求解
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        for status_str in ["unknown", "model_invalid", "feasible", "infeasible", "optimal"]:
            if status == getattr(cp_model, status_str.upper()):
                status_str = status_str.upper()
                break

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"求解失败: {status_str}.")
        # endregion: 约束编程求解

        for old_reg_id, real_reg_id in enumerate(reg_ids):
            new_reg_id = solver.Value(real_reg_id)
            old_reg = regs_to_merge[old_reg_id]
            new_reg = unused_regs[new_reg_id]
            # logger.warning(f"{old_reg.name} is replaced by {new_reg.name}")  ## uger
            self.replace_result_reg(old_reg=old_reg, new_item=new_reg)

    def compile(self) -> list[SMT96]:
        """得到 SMT96 语句

        >>> import brainpy as bp
        >>> funcs = {"V": bp.neurons.LIF(256).derivative}
        >>> compiler = SMT96Compiler(funcs=funcs)
        >>> compiler.compile()
        [R2 = 0.00 - R0, NOP, NOP, NOP, NOP, R2 = 0.00 + R2, NOP, NOP, NOP, NOP, R3 = 1.00 * R1, NOP, NOP, NOP, NOP, R2 = R2 + R3, NOP, NOP, NOP, NOP, R2 = 0.10 * R2, NOP, NOP, NOP, NOP]

        Returns:
            list[SMT96]: SMT 语句
        """

        if self._compiled:
            return self.smt96_results

        # stableHLO 变换成 SMT 96 语句支持的操作, 结果为虚拟寄存器
        self.update_stablehlo_statements()

        # 使用预定义的结果寄存器存储运算结果
        for return_id, name in self.return_names.items():
            if any(name == v.alias for v in self.func_args.values()):
                continue  # 跳过既是输入又是输出的寄存器, 因为运算过程中不能更新输入寄存器
            for predefined_reg in self.regs.used_regs:
                # predefined_regs 中的寄存器只写了 alias 不写 as_arg 或 as_return
                if predefined_reg.alias != name:
                    continue
                if self.stmt_results[return_id] == predefined_reg:
                    continue
                self.replace_result_reg(old_reg=self.stmt_results[return_id], new_item=predefined_reg)
                break

        # 释放没被使用的输入寄存器
        for reg in self.regs.used_regs:
            if reg.used_by == {-1}:
                self.regs.release(reg)

        # 合并虚拟寄存器
        self.merge_regs()

        # 结果寄存器添加名称
        for stmt_id, name in self.return_names.items():
            self.stmt_results[stmt_id].as_return = name
            self.stmt_results[stmt_id].alias = name

        self.smt96_results = []

        for stmt_id, stmts in self.stablehlo_statements.items():
            for stmt in stmts:
                cmd_method = getattr(self.smt_factory, stmt.cmd, None)
                if cmd_method is None:
                    raise NotImplementedError(f"暂不支持指令 {stmt.cmd}")
                a, b = stmt.operands[:2]
                c = self.stmt_results[stmt_id]
                self.smt96_results += cmd_method(a=a, b=b, c=c)
                if stmt.cmd in ["add", "multiply", "subtract"]:
                    self.smt96_results.extend([NOP()] * 4)

        # 1. 通过 as_return 成员找到函数结果寄存器
        # 2. 更新其函数输入寄存器 (相同别名的寄存器)
        for result_reg in filter(lambda r: r.as_return is not None, self.regs.used_regs):
            if result_reg.as_return == "delta_V":  # 跳过 V 寄存器
                continue
            for arg_reg in self.regs.used_regs:
                if result_reg == arg_reg:
                    continue
                if arg_reg.alias == result_reg.as_return:
                    self.smt96_results += self.smt_factory.move(src=result_reg, dst=arg_reg)
                    break

        # 更新 V 的值
        if any(r.as_return == "delta_V" for r in self.regs.used_regs):
            self.regs.get_unused_reg().update(alias="new_V")
            update_v = f"""
            69: R_new_V = R_V + R_delta_V
            70: NOP
            71: NOP
            72: NOP
            73: NOP
            74: R_new_V = R_new_V + {self.constants['I_x']}
            75: NOP
            76: NOP
            77: NOP
            78: NOP
            79: R_V_DIFF_REG0 = {self.constants['V_th']} - R_new_V  // V_diff = V_th - new_V
            80: NOP
            81: NOP
            82: NOP
            83: NOP
            84: R_V = R_new_V, R_CTRL_PULSE = V_SET  // V = new_V if V < V_th else V
            85: R_CTRL_PULSE = SPIKE 
            86: R_CTRL_PULSE = T_SET
            87: R_V_DIFF_REG0 = 0, R_CTRL_PULSE = 0 // reset V_diff
            """
            self.smt96_results += SMT96.create_from_expr(update_v, regs=self.regs)

        return self.smt96_results

    @classmethod
    def compile_all(
        cls,
        funcs: dict[str, Callable],
        constants: dict[str, float],
        predefined_regs: dict[str, str] = None,
    ) -> tuple[SMT96Compiler, SMT96Compiler, list[SMT96]]:
        """编译所有函数.

        0. 如果有运算 I 的函数, 使用一个编译器对象编译此函数
        1. 使用一个编译器对象编译其他函数

        Example:
            >>> import brainpy as bp
            >>> funcs = {"V": bp.neurons.LIF(
            ...     256,
            ...     V_rest=-52,
            ...     V_th=-67,
            ...     V_reset=-22,
            ...     tau=31,
            ...     tau_ref=77,
            ...     method="exp_auto",
            ...     V_initializer=bp.init.Normal(-55.0, 2.0),
            ... ).derivative}
            >>> i_compiler, v_compiler, statements = SMT96Compiler.compile_all(funcs=funcs)
            >>> statements[0]
            R_CTRL_LEVEL = 1, R_ZERO_REG = 0
            >>> statements[1]
            R_STEP_REG = 0, R_PHASE = 0
            >>> statements[2]
            R_CHIP_NPU_ID = 0, R_NEU_NUMS = 1024

        Args:
            funcs (dict[str, Callable]): 需要编译的函数返回值名称和函数体,
                如果返回值名称第一个字母是 I, 则会被认为是 I 函数并改名为 I
            constants (dict[str, float]): 常数, e.g. {"V_th": -50}
            predefined_regs (dict[str, str], Optional): 预定义的寄存器, e.g.
                {"V": "R3"}. 默认为 None.

        Returns:
            tuple[SMT96Compiler, SMT96Compiler, list[SMT96]]: I 编译器, V 编译器, SMT 96 程序
        """

        predefined_regs = predefined_regs or {}  # 参数或结果名称到寄存器的映射, e.g. {"V": "R3"}

        v_compiler = cls(
            funcs={k: v for k, v in funcs.items() if not k.startswith("I")},
        )
        compilers: list[SMT96Compiler] = [v_compiler]

        if i_func_names := [k for k in funcs if k.startswith("I")]:
            i_compiler = cls(
                funcs={k: funcs[k] for k in i_func_names},
            )
            compilers.insert(0, i_compiler)
        else:
            i_compiler = None

        smt96_results: list[SMT96] = []

        used_regs: set[Register96] = set()  # 占用的寄存器的名字

        for compiler in compilers:
            compiler.constants = constants
            for reg in used_regs:  # 上一个编译器使用的寄存器
                new_reg = compiler.regs[reg.index]
                new_reg.update(**reg.__dict__)
                new_reg.used_by = {-3}

            # 预定义的寄存器
            for reg_alias, reg_name in predefined_regs.items():
                compiler.regs[reg_name].alias = reg_alias
                compiler.regs[reg_name].used_by |= {-2}

            smt96_results += compiler.compile()
            used_regs = set(compiler.regs.used_regs)

        return i_compiler, v_compiler, smt96_results
