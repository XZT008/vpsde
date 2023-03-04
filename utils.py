from os.path import join, dirname, exists
import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def show_samples(samples, fname=None, nrow=8, title='Samples'):
    samples = torch.clip(samples, 0, 1)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()