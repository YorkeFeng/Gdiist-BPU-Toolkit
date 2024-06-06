# pylint: disable=too-many-function-args
"""stablehlo 解析器
"""

import pyparsing as pp
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


def operands_update(tokens: pp.ParseResults) -> pp.ParseResults:
    """根据 operands 数值更新 operand.type.

    Args:
        tokens (pp.ParseResults): Pyparsing result

    Returns:
        pp.ParseResults: Updated Pyparsing result
    """
    for i, opd in enumerate(tokens["operands"]):
        otype = list(opd.as_dict())[0]
        tokens["operands"][i] = {"type": otype, "value": opd[otype]["value"]}
    return tokens


cmd_prefix = Suppress("stablehlo.")

integer = pp.pyparsing_common.integer
number = pp.pyparsing_common.number
sci_real = pp.pyparsing_common.sci_real
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

attribute = Group(name + equal + number("value") + colon + dtype).setParseAction(attribute_value_update)

attributes = Group(Suppress("attributes") + left_cb + delimited_list(attribute, delim=",") + right_cb)("attributes")

module_head = Group(Suppress("module") + at_identifier("at_identifier") + attributes)("module_head")

assignment = Group(name + equal + QuotedString(quote_char='"', esc_char="\\")("value"))

assignments = Group(left_cb + delimited_list(assignment, delim=",") + right_cb)("assignments")

arg_def = Group(Suppress(r"%arg") + integer("index") + colon + dtype + assignments("info"))
arg_defs = Group(left_p + delimited_list(arg_def, delim=",") + right_p)("arg_defs")

# '  func.func public @main(%arg0: tensor<i32> {jax.arg_info = "V", mhlo.sharding = "{replicated}"}, %arg1: tensor<i32> {jax.arg_info = "I", mhlo.sharding = "{replicated}"}, %arg2: tensor<i32> {jax.arg_info = "y", mhlo.sharding = "{replicated}"}, %arg3: tensor<i32> {jax.arg_info = "z", mhlo.sharding = "{replicated}"}) -> (tensor<i32> {jax.result_info = ""}) {'
# '  func.func public @main(%arg0: tensor<i32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<i32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<i32> {mhlo.sharding = "{replicated}"}) -> (tensor<f32>, tensor<f32>) {'

dtypes = delimited_list(dtype, delim=",")("dtypes")

return_def = Group(
    left_p + dtypes + Optional(left_cb + Group(delimited_list(assignment, delim=","))("info") + right_cb) + right_p
)("return_def")

func_head = Group(
    Suppress("func.func") + Suppress("public") + at_identifier("name") + arg_defs + Suppress("->") + return_def
)("func_head")

operand_func_arg = Group(Suppress("%arg") + integer("value"))("arg_index")
operand_reg_arg = Group(Suppress("%") + integer("value"))("reg_index")
operand_constant = Group(Suppress("dense") + lt + number("value") + gt)("constant")

operand = Group(operand_func_arg | operand_reg_arg | operand_constant)  # ("operand").setParseAction(operand_update)

operands = delimited_list(operand, delim=",")("operands").setParseAction(operands_update)

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


io_dtype = Group(Optional(left_p + dtype("from") + right_p + Suppress("->")) + dtype("to"))("io_dtype").setParseAction(
    io_dtype_update
)
statement = Group(Suppress("%") + integer("reg_index") + equal + cmd_prefix + cmd + operands + colon + io_dtype)

return_statement = Group(Suppress("return") + operands + colon + dtypes("return_dtypes"))("return_statement")

func_body = Group(left_cb + OneOrMore(statement)("statements") + return_statement + right_cb)("func_body")

stablehlo_parser = module_head + left_cb + func_head + func_body + right_cb
