from config import VQSDE_CONFIG
from vpsde import VQSDE
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


if __name__ == '__main__':
    config = VQSDE_CONFIG()
    train_set = MNIST('./mnist/', train=True, transform=transforms.ToTensor(), download=True)
    test_set = MNIST('./mnist/', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=True, num_workers=6)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir)
    early_stop_callback = EarlyStopping(monitor="BPD", min_delta=0.00, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=1, logger=tb_logger, callbacks=[early_stop_callback])
    model = VQSDE(config)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


