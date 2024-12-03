#%%
import os
from torchvision.transforms import v2
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, num_workers=4, image_size=224, download=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.download = download # Download the dataset if not found

        # Define the transformations for training and validation
        self.train_transforms = v2.Compose([
            v2.RandomResizedCrop(self.image_size),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transforms = v2.Compose([
            v2.Resize(int(self.image_size * 1.14)),
            v2.CenterCrop(self.image_size),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def num_classes(self):
        return 10  # Imagenette has 10 classes

    def prepare_data(self):
        # This method is called only from a single GPU.
        # It downloads the dataset if it doesn't exist already.
        if os.path.exists(os.path.join(self.data_dir, 'imagenette2')):
            self.download = False
        Imagenette(root=self.data_dir, split='train', download=self.download)
        # Imagenette(root=self.data_dir, split='val', download=self.download)

    def setup(self, stage=None):
        # This method is called on every GPU.
        if stage == 'fit' or stage is None:
            self.train_dataset = Imagenette(
                root=self.data_dir,
                split='train',
                transform=self.train_transforms,
                download=False
            )
            self.val_dataset = Imagenette(
                root=self.data_dir,
                split='val',
                transform=self.val_transforms,
                download=False
            )

        if stage == 'test' or stage is None:
            # Using validation set as test set for demonstration.
            self.test_dataset = Imagenette(
                root=self.data_dir,
                split='val',
                transform=self.val_transforms,
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
    batch_size = 32
    num_workers = 4
    image_size = 224
    download = True

    # Initialize the DataModule
    dm = ImagenetteDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, image_size=image_size, download=download)
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
