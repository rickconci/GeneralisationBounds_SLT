from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as L
import torchvision
import torchvision.transforms as transforms
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DataModule(L.LightningDataModule):
    def __init__(self, dataset_name, batch_size=32, random_label_fraction=None, noise_image_fraction=None, train_subset_fraction=0.1, val_subset_fraction=0.1):
        super().__init__()
        self.train_subset_fraction = train_subset_fraction
        self.val_subset_fraction = val_subset_fraction
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        
        self.noise_image_fraction = noise_image_fraction
        self.random_label_fraction = random_label_fraction
        

        if dataset_name == 'CIFAR10':
            self.num_classes = 10
            # CIFAR-10 transformations for resizing and normalization
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
                transforms.CenterCrop(28),  # Crops from the center to 28x28
                transforms.Lambda(per_image_whitening)  # Per-image whitening
            ])
        elif dataset_name == 'ImageNet':
            self.num_classes = 1000
            # ImageNet transformations for resizing and normalization
            self.transform = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])  

    
    def setup(self, stage=None):
        if self.dataset_name == 'CIFAR10':            
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
            if self.random_label_fraction is not None:
                self.randomize_labels()

            if self.noise_image_fraction is not None:
                self.noisy_images()

        elif self.dataset_name == 'ImageNet':
            # Load ImageNet dataset
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
            if self.random_label_fraction is not None:
                self.randomize_labels()

            if self.noise_image_fraction is not None:
                self.noisy_images()

    def randomize_labels(self):
        """Randomize a percentage of labels in the training dataset based on random_label_fraction"""

        if isinstance(self.train, Subset):
            # Access the underlying dataset and indices
            dataset = self.train.dataset
            indices = self.train.indices
        else:
            # If self.train is not a Subset, use it directly
            dataset = self.train
            indices = range(len(self.train))
        
        assert 0.0 <= self.random_label_fraction <= 1.0, "random_label_fraction must be between 0.0 and 1.0"
        # Generate random labels for the subset
        num_samples = len(indices)
        num_randomized = int(num_samples * self.random_label_fraction)
        randomized_indices = np.random.choice(indices, num_randomized, replace=False)
        random_labels = np.random.randint(0, self.num_classes, num_randomized)
        
        # Overwrite the labels in the original dataset at the selected indices
        for idx, new_label in zip(randomized_indices, random_labels):
            dataset.targets[idx] = new_label


    def noisy_images(self):
        """
        Add random noise to the training images in their normalized space.

        The `noise_image_fraction` determines the amount of noise added:
            - 0.0: Original image (no noise).
            - 1.0: Pure Gaussian noise.
            - Values in between: Interpolation between the original image and Gaussian noise.
        """
        if self.noise_image_fraction is not None:
            assert 0.0 <= self.noise_image_fraction <= 1.0, "noise_image_fraction must be between 0.0 and 1.0"

            # Access the training data (already transformed to tensors and normalized)
            images = torch.stack([self.transform(image) for image in self.train.data])
            
            # Generate Gaussian noise in the normalized space
            gaussian_noise = torch.randn_like(images)

            # Interpolate between original and Gaussian noise
            noisy_images = (1 - self.noise_image_fraction) * images + self.noise_image_fraction * gaussian_noise

            # Clip values to a reasonable range for normalized data
            noisy_images = torch.clamp(noisy_images, -1.0, 1.0)  # Normalized images are often in [-1, 1]

            # Replace the dataset's data with noisy images
            self.train.data = noisy_images
    
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=0, persistent_workers=False)




def per_image_whitening(x):
    return (x - x.mean()) / (x.std() + 1e-5)