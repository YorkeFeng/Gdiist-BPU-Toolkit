"""硬件配置信息"""

from dataclasses import dataclass


from ..common.smt_96_base import IBinary


@dataclass
class HardwareSpec:
    """硬件配置信息
    """

    neuron_number: IBinary = IBinary(0, 11)
    """每一个 NPU 中的神经元数量"""

    floating_fix_number_config: IBinary = IBinary(0, 2)
    """浮点数定点数配置"""

    rounding_mode: IBinary = IBinary(0, 4)
    """舍入模式"""

    spike_rate: IBinary = IBinary(0, 15)
    """随机 spike 发放率"""

    @property
    def cfg_bin(self) -> str:
        """硬件配置信息的二进制字符串

        Returns:
            str: 硬件配置信息的二进制字符串
        """
        result = ""
        result += self.spike_rate.bin_value
        result += self.rounding_mode.bin_value
        result += self.floating_fix_number_config.bin_value
        result += self.neuron_number.bin_value
        return result
