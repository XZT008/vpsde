import numpy as np
from network import UNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from utils import savefig, show_samples


class VQSDE(nn.Module):
    def __init__(self, num_chann=1, img_height=28, T=1.0, beta_min=0.1, beta_max=20, eps=1e-5, N=1000, M=10,
                 sample_eps=1e-3, snr=0.16):
        super(VQSDE, self).__init__()
        self.score_func = UNet(input_channels=num_chann,
                               input_height=img_height,
                               ch=32,
                               ch_mult=(1, 2, 2),
                               num_res_blocks=2,
                               attn_resolutions=(16,),
                               resamp_with_conv=True, )
        self.c = num_chann
        self.img_height = img_height
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps
        self.N = N
        self.M = M
        self.sample_eps = sample_eps
        self.snr = snr

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
    def rsde(self, x, t, probability_flow=False):
        drift, diffusion = self.sde(x, t)
        score = self.score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if probability_flow else 1.)
        diffusion = 0. if probability_flow else diffusion
        return drift, diffusion

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
    def pc_sample(self, batch_size=64, use_corrector=True):
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


if __name__ == '__main__':
    """
    model = VQSDE().cuda()
    batch_x = torch.randn(10, 1, 28, 28).cuda()
    dsm_loss = model.dsm_loss(batch_x)
    """


    dataset = MNIST('./mnist/', train=True, transform=transforms.ToTensor(), download=True)

    data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    model = VQSDE().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 50

    for epoch in tqdm(range(epochs)):
        avg_loss = 0.
        num_items = 0
        for x, _ in data_loader:
            x = x.cuda()
            loss = model.dsm_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        print('Average Loss: {:5f}'.format(avg_loss / num_items))
        if epoch > 4 and epoch % 5 == 0:
            samples = model.pc_sample().detach().cpu()
            show_samples(samples, fname=f'./samples/{epoch}.png', nrow=8, title='Samples')




    print()

