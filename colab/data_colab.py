from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as L
import torchvision
import torchvision.transforms as transforms
import numpy as np
#import ssl


class DataModule(L.LightningDataModule):
    def __init__(self, dataset_name, batch_size=32, random_labels=False, random_label_perc=0.1, noisy_image=False, noise_image_perc=0.1, train_subset_fraction=0.1, val_subset_fraction=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.random_labels = random_labels

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

        self.train_subset_fraction = train_subset_fraction
        self.val_subset_fraction = val_subset_fraction

    
    def setup(self, stage=None):
        if self.dataset_name == 'CIFAR10':            
            #ssl._create_default_https_context = ssl._create_unverified_context

            # Load CIFAR-10 dataset
            self.train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.val_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
            
            # Split the validation set into validation and test sets by 90% and 10% respectively    
            self.val = Subset(self.val_test, range(0, int(0.9 * len(self.val_test))))
            self.test = Subset(self.val_test, range(int(0.9 * len(self.val_test)), len(self.val_test)))

            # Create a smaller subset of the training dataset if specified
            if self.train_subset_fraction is not None:
                indices = np.random.choice(len(self.train), int(self.train_subset_fraction * len(self.train)), replace=False)  # Random subset
                self.train = Subset(self.train, indices) 

            # Create a smaller subset of the validation dataset if specified
            if self.val_subset_fraction is not None:
                val_indices = np.random.choice(len(self.val), int(self.val_subset_fraction * len(self.val)), replace=False)  # Random subset
                self.val = Subset(self.val, val_indices)
            
            # Apply random labels if specified
            if self.random_labels:
                self.randomize_labels()

        elif self.dataset_name == 'ImageNet':
            # Load ImageNet dataset

            #ssl._create_default_https_context = ssl._create_unverified_context
            self.train = torchvision.datasets.ImageNet(root='./data', split='train', transform=self.transform)
            self.val_test = torchvision.datasets.ImageNet(root='./data', split='val', transform=self.transform)
            # Split the validation set into validation and test sets by 90% and 10% respectively    
            self.val = Subset(self.val_test, range(0, int(0.9 * len(self.val_test))))
            self.test = Subset(self.val_test, range(int(0.9 * len(self.val_test)), len(self.val_test)))

            if self.train_subset_fraction is not None:
                indices = np.random.choice(len(self.train), int(self.train_subset_fraction * len(self.train)), replace=False)  # Random subset
                self.train = Subset(self.train, indices) 

            # Create a smaller subset of the validation dataset if specified
            if self.val_subset_fraction is not None:
                val_indices = np.random.choice(len(self.val), int(self.val_subset_fraction * len(self.val)), replace=False)  # Random subset
                self.val = Subset(self.val, val_indices)

            # Apply random labels if specified
            if self.random_labels:
                self.randomize_labels()

    def randomize_labels(self):
        """Randomize all labels in the training dataset."""
        if isinstance(self.train, Subset):
            # Access the underlying dataset and indices
            dataset = self.train.dataset
            indices = self.train.indices
        else:
            # If self.train is not a Subset, use it directly
            dataset = self.train
            indices = range(len(self.train))
        
        # Generate random labels for the subset
        num_samples = len(indices)
        random_labels = np.random.randint(0, self.num_classes, num_samples)
        
        # Overwrite the labels in the original dataset at the specified indices
        for idx, new_label in zip(indices, random_labels):
            dataset.targets[idx] = new_label

    #def random_labels(self, random_label_perc):
    #    pass

    def noisy_images(self, noise_image_perc):
        pass
    
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
