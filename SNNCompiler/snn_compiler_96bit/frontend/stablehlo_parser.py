# pylint: disable=too-many-function-args
# pylint: disable=line-too-long
"""Stablehlo IR 解析器

Stablehlo IR 是 JAX 编译器的中间表示, 用于描述 JAX 函数的计算图, e.g

```plain
module @jit_func_for_ir attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
func.func public @main(%arg0: tensor<i32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<i32> {mhlo.sharding = "{replicated}"}) -> (tensor<i32> {jax.result_info = ""}) {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i32>
    %1 = stablehlo.constant dense<42> : tensor<i32>
    %2 = stablehlo.multiply %1, %0 : tensor<i32>
    %3 = stablehlo.add %2, %arg1 : tensor<i32>
    return %3 : tensor<i32>
}
}
```
"""
# pylint: enable=line-too-long

from __future__ import annotations

import inspect
from ctypes import Union
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import jax
import pyparsing as pp
from addict import Dict as AttrDict
from brainpy._src.integrators import JointEq
from pyparsing import (
    Group,
    Keyword,
    OneOrMore,
    Optional,
    QuotedString,
    Suppress,
    Word,
    alphanums,
    delimited_list,
)
from strenum import StrEnum

from ..common.smt_96_reg import RegOrConstant


class StableHLOParser:
    """StableHLO 解析器


    Example:
        >>> import inspect
        >>> import jax
        >>> def func_for_ir(V, V_rest):
        ...     '''测试函数'''
        ...     return 42 * (V - V_rest) + V_rest
        >>> func_jit = jax.jit(func_for_ir)
        >>> func_sig = inspect.signature(func_for_ir)
        >>> func_sig
        <Signature (V, V_rest)>
        >>> arg_count = len(str(func_sig).split(","))
        >>> lower_arg = list(range(arg_count))
        >>> lower_arg
        [0, 1]
        >>> func_ir = str(func_jit.lower(*lower_arg).compiler_ir(dialect="stablehlo"))
        >>> stablehlo_program = StableHLOParser.parse(func_ir)
        >>> stablehlo_program.module_head.at_identifier
        '@jit_func_for_ir'
        >>> stablehlo_program.func_head.name
        '@main'
        >>> stablehlo_program.func_head.arg_defs[0].index
        0
        >>> stablehlo_program.func_head.arg_defs[0].dtype
        'i32'
        >>> stablehlo_program.func_body.statements[0].reg_index
        0
        >>> stablehlo_program.func_body.statements[0].cmd
        'subtract'
        >>> stablehlo_program.func_body.statements[0].operands[0].type
        'arg_index'
        >>> stablehlo_program.func_body.statements[0].operands[0].value
        0
        >>> stablehlo_program.func_body.statements[0].operands[1].type
        'arg_index'
        >>> stablehlo_program.func_body.statements[0].operands[1].value
        1
        >>> stablehlo_program.func_body.return_statement.operands[0].type
        'reg_index'
        >>> stablehlo_program.func_body.return_statement.operands[0].value
        3
        >>> stablehlo_program.func_body.return_statement.return_dtypes
        ['i32']
    """

    @staticmethod
    # pylint: disable-next=too-many-locals
    def attribute_value_update(tokens: pp.ParseResults) -> pp.ParseResults:
        """根据 attribute dtype 更新 attribute value

        Args:
            tokens (pp.ParseResults): Pyparsing result

        Returns:
            pp.ParseResults: Updated Pyparsing result
        """
        # as_dict
        if tokens[0]["dtype"] == "i32":
            tokens[0]["value"] = int(tokens[0]["value"])

        # as_list
        if tokens[0][-1] == "i32":
            tokens[0][-2] = int(tokens[0][-2])

        return tokens

    @staticmethod
    def operands_update(tokens: pp.ParseResults) -> pp.ParseResults:
        """根据 operands 数值更新 operand.type

        Args:
            tokens (pp.ParseResults): Pyparsing result

        Returns:
            pp.ParseResults: Updated Pyparsing result
        """
        for i, opd in enumerate(tokens["operands"]):
            otype = list(opd.as_dict())[0]
            tokens["operands"][i] = {"type": otype, "value": opd[otype]["value"]}
        return tokens

    @staticmethod
    def io_dtype_update(tokens: pp.ParseResults) -> pp.ParseResults:
        """根据 io 数值更新 io.from 和 io.to

        Args:
            tokens (pp.ParseResults): Pyparsing result

        Returns:
            pp.ParseResults: Updated Pyparsing result
        """
        if "from" in tokens["io_dtype"]:
            tokens["io_dtype"]["from"] = tokens["io_dtype"]["from"][0]
        tokens["io_dtype"]["to"] = tokens["io_dtype"]["to"][0]
        del tokens["io_dtype"]["dtype"]
        return tokens

    @classmethod
    # pylint: disable-next=too-many-locals
    def get_stablehlo_parser(cls) -> pp.ParserElement:
        """返回 Stablehlo 解析器

        Returns:
            pp.ParserElement: Stablehlo 解析器
        """

        cmd_prefix = Suppress("stablehlo.")

        integer = pp.pyparsing_common.integer
        number = pp.pyparsing_common.number  # 包括科学计数法
        identifier = pp.pyparsing_common.identifier
        name = Group(delimited_list(identifier, delim="."))("name")
        at_identifier = Word("@", alphanums + "_")
        colon = Suppress(":")
        equal = Suppress("=")
        left_cb = Suppress("{")
        right_cb = Suppress("}")
        left_p = Suppress("(")
        right_p = Suppress(")")
        lt = Suppress("<")
        gt = Suppress(">")
        dtype = Optional(Suppress("tensor")) + Optional(lt) + identifier("dtype") + Optional(gt)

        attribute = Group(name + equal + number("value") + colon + dtype).set_parse_action(cls.attribute_value_update)

        attributes = Group(Suppress("attributes") + left_cb + delimited_list(attribute, delim=",") + right_cb)(
            "attributes"
        )

        module_head = Group(Suppress("module") + at_identifier("at_identifier") + attributes)("module_head")

        assignment = Group(name + equal + QuotedString(quote_char='"', esc_char="\\")("value"))

        assignments = Group(left_cb + delimited_list(assignment, delim=",") + right_cb)("assignments")

        arg_def = Group(Suppress(r"%arg") + integer("index") + colon + dtype + assignments("info"))
        arg_defs = Group(left_p + delimited_list(arg_def, delim=",") + right_p)("arg_defs")

        dtypes = delimited_list(dtype, delim=",")("dtypes")

        return_def = Group(
            left_p
            + dtypes
            + Optional(left_cb + Group(delimited_list(assignment, delim=","))("info") + right_cb)
            + right_p
        )("return_def")

        func_head = Group(
            Suppress("func.func") + Suppress("public") + at_identifier("name") + arg_defs + Suppress("->") + return_def
        )("func_head")

        operand_func_arg = Group(Suppress("%arg") + integer("value"))("arg_index")
        operand_reg_arg = Group(Suppress("%") + integer("value"))("reg_index")
        operand_constant = Group(Suppress("dense") + lt + number("value") + gt)("constant")

        operand = Group(operand_func_arg | operand_reg_arg | operand_constant)
        operands = delimited_list(operand, delim=",")("operands").set_parse_action(cls.operands_update)

        cmd = (
            Keyword("negate")
            | Keyword("convert")
            | Keyword("constant")
            | Keyword("add")
            | Keyword("subtract")
            | Keyword("multiply")
            | Keyword("divide")
            | Keyword("power")
        )("cmd")

        from_dtype = Optional(left_p + dtype("from") + right_p + Suppress("->"))
        io_dtype = Group(from_dtype + dtype("to"))("io_dtype").set_parse_action(cls.io_dtype_update)
        statement = Group(Suppress("%") + integer("reg_index") + equal + cmd_prefix + cmd + operands + colon + io_dtype)
        return_statement = Group(Suppress("return") + operands + colon + dtypes("return_dtypes"))("return_statement")
        func_body = Group(left_cb + OneOrMore(statement)("statements") + return_statement + right_cb)("func_body")

        return module_head + left_cb + func_head + func_body + right_cb

    @classmethod
    def parse(cls, ir_str: str) -> dict:
        """解析 Stablehlo IR 字符串

        Args:
            ir_str (str): Stablehlo IR 字符串

        Returns:
            dict: 解析结果
        """
        stablehlo_parser = cls.get_stablehlo_parser()
        result = AttrDict(stablehlo_parser.parseString(ir_str).as_dict())
        return result


class SupportedOP(StrEnum):
    """当前支持的 StableHLO 命令类型, 在 `StableHLOStatement` 中使用

    - negate: 取负值
    - convert: 类型转换
    - constant: 常数
    - add: 加法
    - subtract: 减法
    - multiply: 乘法
    - divide: 除法
    - power: 指数
    """

    negate = "negate"
    """取负值
    """

    convert = "convert"
    """类型转换
    """

    constant = "constant"
    """常数
    """

    add = "add"
    """加法
    """

    subtract = "subtract"
    """减法
    """

    multiply = "multiply"
    """乘法
    """

    divide = "divide"
    """除法
    """

    power = "power"
    """指数

    - 只支持非负整数常数指数
    - 使用乘法实现运算结果的指数
    - 使用预处理实现常数的指数

    """


@dataclass
class StableHLOStatement:
    """StableHLO 语句, 操作数可以是寄存器对象或者整数
    `StableHLOStatement` 被使用在编译前与编译中

    - 编译前: 保存 StableHLO 语句
        - `reg_index` 为结果寄存器索引
        - `operands` 为字符串列表, e.g. ["arg_index: V", "constant: 42.0"]
    - 编译中: 保存涉及到的寄存器对象
        - `reg_index` 为结果寄存器对象或常数
        - `operands` 为结果寄存器或常数列表
    """

    reg_index: RegOrConstant
    """运算结果
    - `Register96`: 寄存器对象
    - `int`: 寄存器索引 ()
    - `float`: 报错
    """

    cmd: SupportedOP
    """命令类型
    """

    operands: list[Union[RegOrConstant, str]]
    """操作数

    - `Register96`: 函数输入寄存器
    - `float`: 函数输入常数
    - `str`: 还没有进入编译阶段的操作数, e.g.
        - `arg_index: V` 表示函数输入 `V`
        - `constant: 42.0` 表示常数 42.0
            - `StableHLOStatement` 中常数只可能是浮点数
        - `reg_index: 3` 表示第 3 个语句的运算结果
    """


class StableHLOProgram(AttrDict):
    """解析过的 StableHLO 程序

    - 字典成员
        - `module_head`: 函数信息
        - `func_head`: 函数头
        - `func_body`: 函数体

    - 类成员
        - `func_group`: 函数组对象
        - `func_arg_names`: 函数参数名称
        - `statement_list`: StableHLO 指令列表
        - `return_list`: StableHLO 返回值列表

    >>> import inspect
    >>> import jax
    >>> def func_for_ir(V, V_rest):
    ...     '''测试函数'''
    ...     return 42 * (V - V_rest) + V_rest
    >>> prog = StableHLOProgram.load_one_func(func_for_ir)
    >>> prog.module_head.at_identifier
    '@jit_func_for_ir'
    >>> prog.func_head.name
    '@main'
    >>> prog.func_group
    {}
    """

    func_group: FuncGroup
    """函数组对象
    """

    raw_program: list[str]
    """原始程序
    """

    @classmethod
    def load_one_func(cls, func: Callable) -> StableHLOProgram:
        """从一个函数得到 `StableHLOProgram`

        Example:
            >>> import inspect
            >>> import jax
            >>> def func_for_ir(V, V_rest):
            ...     '''测试函数'''
            ...     return 42 * (V - V_rest) + V_rest
            >>> prog = StableHLOProgram.load_one_func(func_for_ir)
            >>> prog.module_head.at_identifier
            '@jit_func_for_ir'
            >>> prog.func_head.name
            '@main'

        Args:
            func (Callable): 函数

        Returns:
            StableHLOProgram: 解析过的 stablehlo IR
        """
        func_sig = inspect.signature(func)
        arg_names = str(func_sig).strip("()").split(",")
        lower_arg = list(range(len(arg_names)))
        func_jit = jax.jit(func, keep_unused=True)
        func_ir = str(func_jit.lower(*lower_arg).compiler_ir(dialect="stablehlo"))
        result = StableHLOProgram(StableHLOParser.parse(func_ir))
        del result.func_body.return_statement["dtype"]

        # region: jax>=0.4.14 取消了 jax.arg_info 属性, 这里添加进去
        for arg_def, arg_name in zip(result.func_head.arg_defs, arg_names):
            for info in arg_def.info:
                if info.name == ["jax", "arg_info"]:
                    if info.value.startswith("I"):
                        info.value = "I"  # 所有以 I 开头的参数都强制改名为 I
                    break  # jax.arg_info 已经存在
            else:
                arg_name = arg_name.strip()
                if arg_name.startswith("I"):  # 所有以 I 开头的参数都强制改名为 I
                    arg_name = "I"
                arg_def.info.append(AttrDict(name=["jax", "arg_info"], value=arg_name))
        # endregion

        return result

    @classmethod
    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def load(cls, func: dict[str, Callable]) -> StableHLOProgram:
        """从一组函数得到 `StableHLOProgram`

        **函数会被解析 2 次, `JointEq` 只会被解析 1 次**

        1. 按顺序记录所有输入
        2. 按顺序合并所有 StableHLO 语句
        3. 按顺序合并所有返回结果

        Example:
            >>> import inspect
            >>> import jax
            >>> import brainpy as bp
            >>> funcs = {"HindmarshRose": bp.neurons.HindmarshRose(256).derivative}
            >>> prog = StableHLOProgram.load(funcs)
            >>> prog.func_group[0].args
            ['V', 't', 'y', 'z', 'I_ext']
            >>> prog.func_group[0].body
            <bound method HindmarshRose.dV ...>
            >>> prog.func_group[0].returns
            'V'
            >>> prog.func_group[1].args
            ['y', 't', 'V']
            >>> prog.func_group[1].body
            <bound method HindmarshRose.dy ...>
            >>> prog.func_group[1].returns
            'y'
            >>> prog.func_group[2].args
            ['z', 't', 'V']
            >>> prog.func_group[2].body
            <bound method HindmarshRose.dz ...>
            >>> prog.func_group[2].returns
            'z'

        Args:
            func (dict[str, Callable]): 函数返回变量名称 -> 函数

        Returns:
            StableHLOProgram: 解析过的 stablehlo IR
        """

        result = StableHLOProgram()
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
            stablehlo_program = StableHLOProgram.load_one_func(one_func.body)
            module_head = module_head or stablehlo_program.module_head

            old_args = stablehlo_program.func_arg_names

            for one_arg_def in stablehlo_program.func_head.arg_defs:
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

            # 改变函数输入编号为函数输入名称
            for stmt in stablehlo_program.func_body.statements:
                stmt.reg_index = len(func_body.statements)
                for opr in stmt.operands:
                    if opr.type == "reg_index":
                        opr.value += reg_index_base
                    elif opr.type == "arg_index":
                        opr.value = old_args[opr.value]
                func_body.statements.append(stmt)

            reg_index_base = len(func_body.statements)

            if func_body.return_statement:
                func_body.return_statement.return_dtypes += stablehlo_program.func_body.return_statement.return_dtypes
                stablehlo_program.func_body.return_statement.operands[0].value = func_body.statements[-1].reg_index
                func_body.return_statement.operands += stablehlo_program.func_body.return_statement.operands
            else:
                func_body.return_statement = stablehlo_program.func_body.return_statement

            func_body.return_statement.operands[-1].name = one_func.returns

        # 合并后的函数名字为 @jit_ 加所有返回值的名字
        module_head.at_identifier = "@jit_" + ("_".join(result.func_group.returns))

        result.module_head = module_head
        result.func_head = func_head
        result.func_body = func_body

        # 删除不需要的 dtype 信息
        if "dtype" in result.func_body.return_statement:
            del result.func_body.return_statement["dtype"]

        name2index = {name: index for index, name in result.func_arg_names.items()}
        for stmt in result.func_body.statements:
            for opr in stmt.operands:
                if opr.type != "arg_index":
                    continue
                opr.value = name2index[opr.value]

        return result

    @cached_property
    def func_arg_names(self) -> AttrDict[int, str]:
        """得到输入函数的参数名称

        Example:
            >>> import inspect
            >>> import jax
            >>> def func_for_ir(V, V_rest):
            ...     '''测试函数'''
            ...     return 42 * (V - V_rest) + V_rest
            >>> prog = StableHLOProgram.load_one_func(func_for_ir)
            >>> prog.module_head.at_identifier
            '@jit_func_for_ir'
            >>> prog.func_head.name
            '@main'
            >>> prog.func_group
            {}

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

    @cached_property
    def statement_list(self) -> list[StableHLOStatement]:
        """StableHLO 指令列表. 操作数为:

        - 寄存器对象: 函数输入或者常数
        - 结果寄存器索引: 运算结果

        >>> import inspect
        >>> import jax
        >>> def func_for_ir(V, V_rest):
        ...     '''测试函数'''
        ...     return 42 * (V - V_rest) + V_rest
        >>> program = StableHLOProgram.load_one_func(func_for_ir)
        >>> program.func_arg_names
        {0: 'V', 1: 'V_rest'}
        >>> program.statement_list[0].reg_index
        0
        >>> program.statement_list[0].cmd
        'subtract'
        >>> program.statement_list[0].operands[0]
        'arg_index: 0'
        >>> program.statement_list[0].operands[1]
        'arg_index: 1'
        >>> program.statement_list[3].reg_index
        3

        Returns:
            list[StableHLOStatement]: IR 指令列表
        """
        result: list[StableHLOStatement] = []
        for stmt_id, raw_stmt in enumerate(self.func_body.statements):
            operands = [f"{opr.type}: {opr.value}" for opr in raw_stmt.operands]
            stmt = StableHLOStatement(reg_index=stmt_id, cmd=raw_stmt.cmd, operands=operands)
            result.append(stmt)
        return result

    @property
    def return_list(self) -> list[str]:
        """StableHLO 返回值列表. 操作数为:

        >>> import inspect
        >>> import jax
        >>> def func_for_ir(V, V_rest):
        ...     '''测试函数'''
        ...     return 42 * (V - V_rest) + V_rest
        >>> program = StableHLOProgram.load_one_func(func_for_ir)
        >>> program.return_list
        ['reg_index: 3']

        Returns:
            list[str]: IR 指令列表
        """
        result: list[StableHLOStatement] = []
        for opr in self.func_body.return_statement.operands:
            result.append(f"{opr.type}: {opr.value}")
        return result


@dataclass
class Func:
    """函数模型, 在 `StableHLOProgram` 中使用

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
    def load(cls, func: dict[str, Callable]) -> FuncGroup:
        """从函数对象读取函数模型

        - 使用 `StableHLOProgram.load_one_func` 方法解析单个函数找到输入输出变量名字
        - `JointEq` 可以直接得到输入输出的名字不需要解析

        Args:
            func (dict[str, Callable]): 函数返回值名称 -> 函数或函数组 (`JointEq`)
                如果 func 的值是函数组 (`JointEq`) 则忽略返回值名称

        Returns:
            FuncGroup: 函数组对象
        """
        result = FuncGroup()
        for return_name, one_func in func.items():
            if isinstance(one_func, JointEq):  # 读取函数组信息
                for i, result_vars in enumerate(one_func.vars_in_eqs):
                    result += [Func(args=one_func.args_in_eqs[i], body=one_func.eqs[i], returns=result_vars[0])]
            else:
                stablehlo_program = StableHLOProgram.load_one_func(one_func)
                result += [
                    Func(args=list(stablehlo_program.func_arg_names.values()), body=one_func, returns=return_name)
                ]
        return result


class FuncGroup(list):
    """函数模型列表, 在 `StableHLOProgram` 中使用

    - `args`: 所有函数的参数变量名字
    - `results`: 所有函数的结果变量名字

    >>> FuncGroup([3, 4])
    Traceback (most recent call last):
    ...
    RuntimeError: 3 不是 Func 类型而是 <class 'int'>
    """

    def __init__(self, *args):
        super().__init__(*args)
        for one_func in self:
            if not isinstance(one_func, Func):
                raise RuntimeError(f"{one_func} 不是 Func 类型而是 {type(one_func)}")

    @property
    def args(self) -> list[str]:
        """按顺序得到所有函数的参数, 同名参数只取一个

        Returns:
            list[str]: 所有函数的参数
        """
        result: list[str] = []
        for one_func in self:
            for one_arg in one_func.args:
                if one_arg not in result:
                    result += [one_arg]
        return result

    @property
    def returns(self) -> list[str]:
        """安顺序得到所有函数的结果变量名, 同名结果变量只取一个

        Returns:
            list[str]: 所有函数的结果变量名
        """
        result = []
        for one_func in self:
            if one_func.returns not in result:
                result += [one_func.returns]
        return result


__all__ = ["StableHLOStatement", "StableHLOProgram"]
