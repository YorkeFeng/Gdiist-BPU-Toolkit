#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yukun Feng
# @Date: 2024-05-16

from pathlib import Path

import brainpy as bp
import numpy as np
from brainpy.neurons import LIF
from brainpy.synapses import Exponential
from loguru import logger


class EINet(bp.Network):
    def __init__(self, num_exc, num_inh, connect_prob, method='exp_auto', allow_multi_conn=True, **kwargs):
        super(EINet, self).__init__(**kwargs)

        # 导入神经元数目
        self.num_exc = num_exc
        self.num_inh = num_inh
        self.total = num_exc + num_inh
        self.neuron_scale = self.num_exc / self.total

        # 初始化读取变量
        self.E_v = []
        self.E_input = []
        self.E_spike = []
        self.I_v = []
        self.I_input = []
        self.I_spike = []
        self.E2E_g = []
        self.E2I_g = []
        self.I2E_g = []
        self.I2I_g = []

        # neurons
        self.V_reset = -60
        self.V_rest = -60
        self.V_thresh = -50
        self.tau = 20.0
        self.tar_ref = 5.0
        E_pars = dict(V_rest=self.V_rest, V_th=self.V_thresh,
                      V_reset=self.V_reset, tau=self.tau, tau_ref=self.tar_ref)
        I_pars = dict(V_rest=self.V_rest, V_th=self.V_thresh,
                      V_reset=self.V_reset, tau=self.tau, tau_ref=self.tar_ref)
        E = LIF(num_exc, **E_pars, method=method)
        I = LIF(num_inh, **I_pars, method=method)

        # 固定所有膜电位初始值为-60
        E.V.value = bp.math.zeros(num_exc) - self.V_reset
        I.V.value = bp.math.zeros(num_inh) - self.V_reset

        # synapses
        self.E_E = 0.0
        self.I_E = -80.0
        w_e = 0.8187  # excitatory synaptic weight
        w_i = 0.9048  # inhibitory synaptic weight
        E_pars = dict(output=bp.synouts.COBA(E=self.E_E), g_max=w_e, tau=5.)
        I_pars = dict(output=bp.synouts.COBA(E=self.I_E), g_max=w_i, tau=10.)

        # Neurons connect to each other randomly with a connection probability of 2%
        self.E2E = Exponential(E, E, bp.conn.FixedProb(
            prob=connect_prob, allow_multi_conn=allow_multi_conn), **E_pars, method=method)
        self.E2I = Exponential(E, I, bp.conn.FixedProb(
            prob=connect_prob, allow_multi_conn=allow_multi_conn), **E_pars, method=method)
        self.I2E = Exponential(I, E, bp.conn.FixedProb(
            prob=connect_prob, allow_multi_conn=allow_multi_conn), **I_pars, method=method)
        self.I2I = Exponential(I, I, bp.conn.FixedProb(
            prob=connect_prob, allow_multi_conn=allow_multi_conn), **I_pars, method=method)

        self.E = E
        self.I = I

    def update_state(self, tdi, x):
        self.E2E(tdi, x[: self.num_exc, ])
        self.E2I(tdi, x[: self.num_exc, ])
        self.I2E(tdi, x[self.num_exc: self.total, ])
        self.I2I(tdi, x[self.num_exc: self.total, ])
        self.E(tdi)
        self.I(tdi)

        self.E_v.append(self.E.V.value)
        self.E_input.append(self.E.input.value)
        self.E_spike.append(self.E.spike.value)
        self.I_v.append(self.I.V.value)
        self.I_input.append(self.I.input.value)
        self.I_spike.append(self.I.spike.value)
        self.E2E_g.append(self.E2E.g.value)
        self.E2I_g.append(self.E2I.g.value)
        self.I2E_g.append(self.I2E.g.value)
        self.I2I_g.append(self.I2I.g.value)

        # self.E.clear_input()
        # self.I.clear_input()

    def run(self, spike_I, step, download_dir):
        # simulation
        logger.info("Software data running.")
        tdi = {"t": 1., "dt": 1}
        for i in range(step):
            tdi['t'] = i
            self.t = i
            self.update_state(tdi, spike_I[i])

        spike_matrix = np.zeros((step, self.neuron_num))
        V_matrix = np.zeros((step, self.neuron_num))

        download_dir = Path(download_dir)
        download_dir.mkdir(exist_ok=True)
        soft_dir = download_dir / 'soft_data'
        soft_dir.mkdir(exist_ok=True)

        logger.info("Saving...............")
        np.save(download_dir / "soft_data/E2I_g.npy", np.array(self.E2I_g))
        np.save(download_dir / "soft_data/E2E_g.npy", np.array(self.E2E_g))
        np.save(download_dir / "soft_data/I2I_g.npy", np.array(self.I2I_g))
        np.save(download_dir / "soft_data/I2E_g.npy", np.array(self.I2E_g))

        spike_matrix[:, : self.num_exc] = np.array(self.E_spike)
        spike_matrix[:, self.num_exc:] = np.array(self.I_spike)
        V_matrix[:, :self.num_exc] = np.array(self.E_v)
        V_matrix[:, self.num_exc:] = np.array(self.I_v)

        np.save(download_dir / "soft_data/N_spike.npy", spike_matrix)
        np.save(download_dir / "soft_data/N_V.npy", V_matrix)
        logger.info('Software data saved.')
