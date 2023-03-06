import numpy as np
from network import UNet
import torch
from utils import show_samples
from scipy import integrate
from config import VQSDE_CONFIG
import pytorch_lightning as pl


class VQSDE(pl.LightningModule):
    def __init__(self, cfg, tune_cfg=None, if_tune=False):
        super(VQSDE, self).__init__()
        if if_tune is not False:
            assert tune_cfg is not None
        self.if_tune = if_tune
        self.config = cfg
        self.tune_config = tune_cfg
        if not if_tune:
            self.score_func = UNet(input_channels=self.config.num_chann,
                                   input_height=self.config.img_height,
                                   ch=32,
                                   ch_mult=self.config.ch_mult,
                                   num_res_blocks=self.config.num_res_blocks,
                                   attn_resolutions=(16,),
                                   resamp_with_conv=True, )
        else:
            self.score_func = UNet(input_channels=self.config.num_chann,
                                   input_height=self.config.img_height,
                                   ch=32,
                                   ch_mult=tune_cfg['ch_mult'],
                                   num_res_blocks=tune_cfg['num_res_blocks'],
                                   attn_resolutions=(16,),
                                   resamp_with_conv=True, )
        self.c = self.config.num_chann
        self.img_height = self.config.img_height
        self.T = self.config.T
        self.beta_min = self.config.beta_min
        self.beta_max = self.config.beta_max
        self.eps = self.config.eps
        self.N = self.config.N
        self.M = self.config.M
        self.sample_eps = self.config.sample_eps
        self.snr = self.config.snr

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5*beta_t.view(-1, 1, 1, 1)*x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def get_alpha_beta_t(self, t):
        discrete_beta_t = (self.beta_min + t * (self.beta_max - self.beta_min)) / self.N
        alpha_t = 1 - discrete_beta_t
        return alpha_t

    def marginal_prob_mean_std(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff.view(-1, 1, 1, 1)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def perturb(self, x):
        batch_size = x.shape[0]
        t = torch.rand(batch_size).cuda() * (self.T - self.eps) + self.eps
        z = torch.randn_like(x).cuda()
        mean, std = self.marginal_prob_mean_std(x, t)
        x_tilda = mean + std.view(-1, 1, 1, 1) * z
        return x_tilda, t, z, mean, std

    def forward(self, x):
        x_tilda, t, z, mean, std = self.perturb(x)
        normed_score = self.score_func(x_tilda, t) / std.view(-1, 1, 1, 1)
        return normed_score, std, t, z

    def score_fn(self, x, t):
        _, std = self.marginal_prob_mean_std(x, t)
        score = self.score_func(x, t)
        return score / std.view(-1, 1, 1, 1)

    def dsm_loss(self, x):
        normed_score, std, t, z = self(x)
        dsm_loss = torch.mean(torch.sum((normed_score * std.view(-1, 1, 1, 1) + z)**2, dim=(1, 2, 3)))
        return dsm_loss

    # below are sampling related functions

    # reverse of sde
    def rsde(self, x, t):
        drift, diffusion = self.sde(x, t)
        score = self.score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score
        return drift, diffusion

    def rode(self, x, t):
        drift, diffusion = self.sde(x, t)
        score = self.score_fn(x, t)
        drift = drift - 0.5 * diffusion[:, None, None, None] ** 2 * score
        diffusion = 0                   # this is to correspond with rsde
        return drift

    # langevin dynamics
    def corrector(self, x, t):
        alpha = self.get_alpha_beta_t(t)
        for i in range(self.M):
            grad = self.score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
        return x, x_mean

    # euler_maruyama
    def predictor(self, x, t):
        dt = -1. / self.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion.view(-1, 1, 1, 1) * np.sqrt(-dt) * z
        return x, x_mean

    # pc sampler
    def pc_sampler(self, batch_size=64, use_corrector=True):
        with torch.no_grad():
            x = torch.randn(batch_size, self.c, self.img_height, self.img_height).cuda()
            time_steps = torch.linspace(1.0, self.sample_eps, self.N)
            for i in range(self.N):
                t = time_steps[i]
                vec_t = torch.ones(batch_size).cuda() * t
                if use_corrector:
                    x, x_mean = self.corrector(x, vec_t)
                x, x_mean = self.predictor(x, vec_t)

            return x_mean

    # function used by scipy.integrate.solve_ivp
    def ode_func_sampling(self, t, x):
        # convert x from (n, ) ndarray to batch of images in torch tensor
        x = x.reshape((self.samples_batch_size, self.c, self.img_height, self.img_height))
        x = torch.from_numpy(x).cuda().type(torch.float32)
        vec_t = torch.ones(self.samples_batch_size).cuda() * t
        drift = self.rode(x, vec_t)
        return drift.detach().cpu().numpy().reshape((-1,))          # need to return a ndarray of shape(n, )

    def ode_sampler(self, batch_size=64, rtol=1e-5, atol=1e-5, method='RK45'):
        self.samples_batch_size = batch_size
        with torch.no_grad():
            x = torch.randn(batch_size, self.c, self.img_height, self.img_height).cuda()
            solution = integrate.solve_ivp(self.ode_func_sampling, (self.T, self.sample_eps), x.detach().cpu().numpy().reshape((-1,)),
                                           rtol=rtol, atol=atol, method=method)
            x = torch.tensor(solution.y[:, -1]).reshape(batch_size, self.c, self.img_height, self.img_height).cuda().type(torch.float32)
        return x

    def log_pT_xT(self, z):
        N = np.prod(self.c * self.img_height * self.img_height)
        log_prop = N * (np.log(1.0/np.sqrt(2*np.pi))) - 0.5 * torch.sum(z ** 2, dim=(1, 2, 3))
        return log_prop

    def ode_func_likelihood_estimation(self, t, x):
        sample = torch.from_numpy(x[:self.cut_off_pos]).reshape(self.likelihood_data_shape).cuda().type(torch.float32)
        vec_t = torch.ones(sample.shape[0]).cuda() * t
        with torch.enable_grad():
            sample.requires_grad = True
            drift = self.rode(sample, vec_t)
            grad_fn_eps = torch.autograd.grad(torch.sum(drift * self.epsilon), sample)[0]
        sample.requires_grad = False
        with torch.no_grad():
            div = torch.sum(grad_fn_eps * self.epsilon, dim=(1, 2, 3)).reshape(-1).detach().cpu().numpy()
            drift = drift.reshape(-1).detach().cpu().numpy()
        return np.concatenate([drift, div], axis=0)

    def likelihood_estimation(self, data, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
        with torch.no_grad():
            self.epsilon = torch.randn_like(data).cuda()
            self.likelihood_estimation_batch_size = data.shape[0]
            self.cut_off_pos = self.likelihood_estimation_batch_size * self.c * self.img_height * self.img_height
            self.likelihood_data_shape = data.shape

            init = np.concatenate([data.detach().cpu().numpy().reshape((-1,)), np.zeros((self.likelihood_estimation_batch_size,))], axis=0)
            solution = integrate.solve_ivp(self.ode_func_likelihood_estimation, (eps, self.T), init, rtol=rtol, atol=atol, method=method)
            zp = solution.y[:, -1]
            z = torch.from_numpy(zp[:self.cut_off_pos]).reshape(self.likelihood_data_shape).cuda().type(torch.float32)
            delta_logp = torch.from_numpy(zp[self.cut_off_pos:]).cuda().type(torch.float32)
            log_pt_xt = self.log_pT_xT(z)
            log_likelihood = (log_pt_xt + delta_logp).detach().cpu().numpy()
            return log_likelihood

    def bpd_calculation(self, data, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
        log_likelihood = self.likelihood_estimation(data, rtol, atol, method, eps)
        bpd = -log_likelihood / np.log(2)                   # change base from e to 2
        bpd = bpd / (self.c * self.img_height * self.img_height) + 8
        return np.mean(bpd)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.dsm_loss(x)
        self.log('DSM_loss_train', loss)
        return loss

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        self.trainer.save_checkpoint(filepath=f'./checkpoints/{epoch}.ckpt')

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        if batch_idx < self.config.bpd_batch_num:
            x_uniform_deq = (x * 255. + torch.rand_like(x)) / 256.
            if self.if_tune:
                bpd = self.bpd_calculation(x_uniform_deq, rtol=self.config.bpd_rtol, atol=self.config.bpd_atol,
                                           method=self.config.bpd_method, eps=self.tune_config['sampling_eps'])
            else:
                bpd = self.bpd_calculation(x_uniform_deq, rtol=self.config.bpd_rtol, atol=self.config.bpd_atol,
                                           method=self.config.bpd_method, eps=self.config.bpd_eps)
            self.log('BPD', bpd)
        loss = self.dsm_loss(x)
        self.log('DSM_loss_test', loss)
        return loss

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        if self.config.sampler == 'both':
            pc_samples = self.pc_sampler(batch_size=self.config.pc_sample_batch_size,
                                         use_corrector=self.config.use_corrector).detach().cpu()
            ode_samples = self.ode_sampler(batch_size=self.config.ode_sample_batch_size, rtol=self.config.rtol,
                                           atol=self.config.atol, method=self.config.method).detach().cpu()
            show_samples(pc_samples, fname=f'./pc_samples/{epoch}.png')
            show_samples(ode_samples, fname=f'./ode_samples/{epoch}.png')
        elif self.config.sampler == 'pc':
            pc_samples = self.pc_sampler(batch_size=self.config.pc_sample_batch_size,
                                         use_corrector=self.config.use_corrector).detach().cpu()
            show_samples(pc_samples, fname=f'./pc_samples/{epoch}.png')
        elif self.config.sampler == 'ode':
            ode_samples = self.ode_sampler(batch_size=self.config.ode_sample_batch_size, rtol=self.config.rtol,
                                           atol=self.config.atol, method=self.config.method).detach().cpu()
            show_samples(ode_samples, fname=f'./ode_samples/{epoch}.png')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

