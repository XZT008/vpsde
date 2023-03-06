from config import VQSDE_CONFIG
from vpsde import VQSDE
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune


def tune_mnist(tune_cfg, cfg=VQSDE_CONFIG(), num_epochs=10, num_gpus=1):
    model = VQSDE(cfg, tune_cfg, if_tune=True)
    train_set = MNIST('./mnist/', train=True, transform=transforms.ToTensor(), download=True)
    test_set = MNIST('./mnist/', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_set, batch_size=int(tune_cfg['batch_size']), shuffle=True, num_workers=6)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=True, num_workers=6)
    metrics = {'loss': 'DSM_loss_test', 'bpd': 'BPD'}
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    tune_config = {
        'lr': tune.loguniform(1e-5, 1e-2),
        'batch_size': tune.choice([128, 256, 512]),
        'num_res_blocks': tune.choice([1, 2]),
        'sampling_eps': tune.loguniform(1e-5, 1e-3),
        'ch_mult': tune.choice([(1, 2, 2), (1, 2, 4), (1, 2, 8)]),
    }

    trainable = tune.with_parameters(
        tune_mnist,
        cfg=VQSDE_CONFIG(),
        num_epochs=10,
        num_gpus=1)

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        },
        metric="bpd",
        mode="min",
        config=tune_config,
        num_samples=5,
        name="tune_mnist")

    print(analysis.best_config)
