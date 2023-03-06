import os
import numpy


class VQSDE_CONFIG:
    def __init__(self):
        # model param
        self.num_chann = 1
        self.img_height = 28
        self.ch_mult = (1, 2, 2)
        self.num_res_blocks = 2
        self.T = 1.0
        self.beta_min = 0.1
        self.beta_max = 20
        self.eps = 1e-5
        self.N = 1000
        self.M = 10
        self.sample_eps = 1e-3
        self.snr = 0.16

        # pc_sampler param
        self.pc_sample_batch_size = 64
        self.use_corrector = True

        # ode_sampler param
        self.ode_sample_batch_size = 64
        self.rtol = 1e-5
        self.atol = 1e-5
        self.method = 'RK45'

        # which sampler to use, 'ode', 'pc', or 'both'
        self.sampler = 'both'

        # bpd eval param
        self.bpd_rtol = 1e-5
        self.bpd_atol = 1e-5
        self.bpd_method = 'RK45'
        self.bpd_eps = 1e-5
        self.bpd_batch_num = 3

        # logging
        self.log_dir = 'log/'
        # self.checkpoints_dir = './checkpoints/'

