#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yukun Feng
# @Date: 2024-05-21


import time
import warnings

import brainpy.math as bm
import click
import jax
import numpy as np
from loguru import logger

from BrainpyAdapter.EImodel import EINet
from SNN40nm import SNN40nmASIC

warnings.filterwarnings("ignore")


@click.command()
@click.argument('download_dir')
def test(download_dir):
    t0 = time.time()
    label_list = []
    time_list = []

    scope = 128
    T = 0
    spk_ranges = 1.6
    key = jax.random.PRNGKey(1)
    bm.random.seed(42)

    # scope = 2**scope
    neuronScale = 0.5
    total_num = 1024*scope
    connect_prob = 5 / total_num
    ex_num = int(total_num * neuronScale)
    ih_num = int(total_num * neuronScale)

    print(rf'download_dir: {download_dir}')
    net = EINet(ex_num, ih_num, connect_prob,
                method='exp_auto', allow_multi_conn=True)
    x = bm.where(jax.random.normal(key, shape=(
        min(16384, total_num),)) >= spk_ranges, 1, 0)
    I = np.zeros((max(T, 1), total_num))
    I[0][:min(16384, total_num)] = x

    # mode 0,1,2 = no file saved, save spike, save all
    t1 = time.time()
    logger.info('Initial Brainpy Network. Elapsed: %.2f s\n' % (t1-t0))  # 输出
    label_list.append("Init Brainpy")
    time_list.append(t1-t0)

    config_dir = '/home/yorke/gdiist/git/Gdiist-BPU-Toolkit/HardwareConfig/Config_40nmASIC.yaml'
    bpuset = SNN40nmASIC(net, I, config_file=config_dir)
    bpuset.gen_bin_data(download_dir)
    # bpuset.run(T, mode=1, upload_dir=upload_dir, reset = True)


if __name__ == "__main__":
    test()
