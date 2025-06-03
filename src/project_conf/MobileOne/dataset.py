import pytorch_lightning as pl
import numpy as np
import torch

from collections import Counter
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset


class SkinCancerDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size, num_workers):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        # Define transformation for image
        train_data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ])

        test_data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.train_data = ImageFolder(root=self.train_dir, transform=train_data_transform)
        self.val_data_full = ImageFolder(root=self.val_dir, transform=test_data_transform)

        # Shuffle val indices ONCE using fixed seed
        np.random.seed(42)
        val_indices = np.random.permutation(len(self.val_data_full))
        self.val_data = Subset(self.val_data_full, val_indices.tolist())
    
        # self.num_classes = 1 if len(self.val_data.classes) == 2 else len(self.val_data.classes)
        self.num_classes = 1

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=False)
    
    def test_dataloader(self):
        pass
    
    def get_num_classes(self):
        return self.num_classes