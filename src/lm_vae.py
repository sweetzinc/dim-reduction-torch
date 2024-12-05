#%%
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from lightning.pytorch.utilities.model_summary import ModelSummary

from models.base_vae import BaseVAE


class LM_VAE(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super().__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, **kwargs)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        # Forward pass
        results = self.forward(x, labels=labels)
        
        # Compute loss
        train_loss = self.model.loss_function(results,
                                              kld_weight=self.params['kld_weight'],
                                              batch_idx=batch_idx)
        
        # Log training losses
        self.log_dict(
            {f"train_{key}": val.item() for key, val in train_loss.items()},
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        # Forward pass
        results = self.forward(x, labels=labels)

        # Compute loss
        val_loss = self.model.loss_function(results,
                                            kld_weight=self.params['kld_weight'],
                                            batch_idx=batch_idx)
        
        # Log validation losses
        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
   
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

#%%
if __name__ == "__main__":
    from models.vanilla_vae import VanillaVAE
    from data_handling.pl_fashion_mnist import FashionMNISTDataModule

    model_config = {
        'in_channels': 1,
        'latent_dim': 2,
        'hidden_dims': [32, 64, ],
        'width': 28, 'height': 28 }
    lightning_config = {
        'LR': 1e-3,
        'weight_decay': 1e-5,
        'scheduler_gamma': 0.95,
        'kld_weight': 1,}
    data_config = {
        'data_dir': '/mounted_data/downloaded',
        'batch_size': 64,
        'num_workers': 4}

    # Define model
    vae = VanillaVAE(**model_config)
    lightning_module = LM_VAE(vae_model=vae, params=lightning_config)
    
    summary = ModelSummary(model=lightning_module, max_depth=-1)
    print(summary)

    # Initialize DataModule
    dm = FashionMNISTDataModule(**data_config)
    dm.prepare_data(); dm.setup()

    # Set up and run Trainer
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(lightning_module, dm)


# %%
