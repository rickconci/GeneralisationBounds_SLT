from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import lightning as L
import torchvision
import torchvision.transforms as transforms
import numpy as np


class DataModule(L.LightningDataModule):
    def __init__(self, dataset_name, batch_size=32, random_labels=False, random_label_perc=0.1, noisy_image=False, noise_image_perc=0.1, train_subset_size=64, val_subset_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        if dataset_name == 'CIFAR10':
            self.num_classes = 10
            # CIFAR-10 transformations for resizing and normalization
            self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif dataset_name == 'ImageNet':
            self.num_classes = 1000
            # ImageNet transformations for resizing and normalization
            self.transform = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])  

        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size

    
    def setup(self, stage=None):
        if self.dataset_name == 'CIFAR10':
            # Load CIFAR-10 dataset
            self.train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.val_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
            
            # Split the validation set into validation and test sets by 90% and 10% respectively    
            self.val = Subset(self.val_test, range(0, int(0.9 * len(self.val_test))))
            self.test = Subset(self.val_test, range(int(0.9 * len(self.val_test)), len(self.val_test)))

            # Create a smaller subset of the training dataset if specified
            if self.train_subset_size is not None:
                indices = np.random.choice(len(self.train), self.train_subset_size, replace=False)  # Random subset
                self.train = Subset(self.train, indices) 

            # Create a smaller subset of the validation dataset if specified
            if self.val_subset_size is not None:
                val_indices = np.random.choice(len(self.val), self.val_subset_size, replace=False)  # Random subset
                self.val = Subset(self.val, val_indices)

        elif self.dataset_name == 'ImageNet':
            # Load ImageNet dataset
            self.train = torchvision.datasets.ImageNet(root='./data', split='train', transform=self.transform)
            self.val_test = torchvision.datasets.ImageNet(root='./data', split='val', transform=self.transform)
            # Split the validation set into validation and test sets by 90% and 10% respectively    
            self.val = Subset(self.val_test, range(0, int(0.9 * len(self.val_test))))
            self.test = Subset(self.val_test, range(int(0.9 * len(self.val_test)), len(self.val_test)))

    def random_labels(self, random_label_perc):
        pass

    def noisy_images(self, noise_image_perc):
        pass
    
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
