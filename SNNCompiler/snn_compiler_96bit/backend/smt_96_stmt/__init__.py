"""SMT 96 位指令"""

from .fp_op import FP_OP
from .jump_op import JUMP_OP
from .nop import NOP
from .reg_op import REG_OP
from .smt96 import SMT96
from .sram_op import SRAM_OP

__all__ = ["SMT96", "NOP", "SRAM_OP", "REG_OP", "FP_OP", "JUMP_OP"]
