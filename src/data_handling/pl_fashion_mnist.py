#%%
import os
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir='./data',
        batch_size=64,
        num_workers=4,
        download=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        # Transforms for the data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    @property
    def num_classes(self):
        return 10  # FashionMNIST has 10 classes

    def prepare_data(self):
        # Download the dataset if not already present
        FashionMNIST(root=self.data_dir, train=True, download=self.download)
        FashionMNIST(root=self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            full_train_dataset = FashionMNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform,
                download=False
            )
            # Split the dataset into train and validation subsets
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train_dataset, [train_size, val_size]
            )

        if stage == 'test' or stage is None:
            self.test_dataset = FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
                download=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

#%%
if __name__ == '__main__' :
    # Configuration
    data_dir = '/mounted_data/downloaded'  # Directory to save the data
    batch_size = 128
    num_workers = 4
    download = True

    # Initialize the DataModule
    dm = FashionMNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, download=download)
    dm.prepare_data()
    dm.setup()

    # Loaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Test the loaders
    for loader in [train_loader, val_loader, test_loader]:
        for data, target in loader:
            print("Data shape:", data.shape)
            print("Target shape:", target.shape)
            break
        break
# %%
