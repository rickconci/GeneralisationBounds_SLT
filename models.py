import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.parametrizations import weight_norm
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import Accuracy
import lightning as L
from lightning import LightningModule
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.init as init


from utils import max_pixel_sums, eval_rho, our_total_bound


class SparseDeepModel(LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        super(SparseDeepModel, self).__init__()

        if model_name == 'InceptionV3':
            self.model = models.inception_v3(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'AlexNet':
            self.model = models.alexnet(weights=None)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        

        self.criterion = nn.MSELoss() # changed loss function and got error!
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)


        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)
        self.train_acc = self.accuracy(outputs.argmax(dim=1), labels)

        # Log the training loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # Access the training dataloader
        data_loader = self.trainer.datamodule.train_dataloader()
        num_classes = self.hparams.num_classes
        dataset_size = len(data_loader.dataset)
        depth = len([layer for layer in self.model.modules() if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))])

        # Compute the generalization bound using the utility function
        bound = our_total_bound(self, data_loader, num_classes, dataset_size, depth)
        self.log("generalization_bound", bound, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)
        self.val_acc = self.accuracy(outputs.argmax(dim=1), labels)
        
        # Log the validation loss and accuracy
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)  
        self.test_acc = self.accuracy(outputs.argmax(dim=1), labels)

        # Log the test loss and accuracy
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

        # Access the training dataloader
        data_loader = self.trainer.datamodule.train_dataloader()
        num_classes = self.hparams.num_classes
        dataset_size = len(data_loader.dataset)
        depth = len([layer for layer in self.model.modules() if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))])

        # Compute the generalization bound using the utility function
        bound = our_total_bound(self, data_loader, num_classes, dataset_size, depth)
        self.log("generalization_bound", bound, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Define SGD optimizer with weight decay
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer



class ModularCNN(LightningModule):
    def __init__(self, kernel_dict, max_pool_layer_dict, dropout_layer_dict, num_classes, lr, weight_decay, warmup_steps=50, max_steps=200):
        super(ModularCNN, self).__init__()
        # implement model with N number of Conv2d layers followed by potential maxpooling layers, relu activations, dropout layers, and one single final linear layer
        # input size is always 28x28
        # kernel_dict is a nested dictinary that describes the kernel size for each conv layer including the number of output channels for each conv layer. 1st keys are the layer indices. 2nd keys include the following: kernel_size, output_channels, stride, padding
        # max_pool_layer_dict describes the max pool size for each max pool layer and at which layer it is applied
        # dropout_layer_dict describes the dropout rate for each dropout layer and at which layer it is applied

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.num_classes = num_classes

        # Assuming grayscale images; change to 3 if using RGB images
        in_channels = 1
        h, w = 28, 28  # Input image dimensions

        layers = []
        num_layers = max(
            max(kernel_dict.keys(), default=0),
            max(max_pool_layer_dict.keys(), default=0),
            max(dropout_layer_dict.keys(), default=0)
        )

        # Build layers based on the dictionaries
        for layer_idx in range(num_layers + 1):
            # Convolutional Layer
            if layer_idx in kernel_dict:
                layer_params = kernel_dict[layer_idx]
                kernel_size = layer_params['kernel_size']
                out_channels = layer_params['out_channels']
                stride = layer_params['stride']
                padding = layer_params['padding']

                conv_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
                layers.append(conv_layer)
                conv_layer = weight_norm(conv_layer)
                layers.append(nn.ReLU(inplace=True))

                # Update spatial dimensions
                h = (h + 2 * padding - kernel_size) // stride + 1
                w = (w + 2 * padding - kernel_size) // stride + 1

                # Update channels for next layer
                in_channels = out_channels

            # Max Pooling Layer
            if layer_idx in max_pool_layer_dict:
                layer_params = max_pool_layer_dict[layer_idx]
                pool_size = layer_params['pool_size']
                stride = layer_params['stride']
                maxpool_layer = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
                layers.append(maxpool_layer)

                # Update spatial dimensions
                h = (h - pool_size) // stride + 1
                w = (w - pool_size) // stride + 1

            # Dropout Layer
            if layer_idx in dropout_layer_dict:
                p = dropout_layer_dict[layer_idx]
                dropout_layer = nn.Dropout(p=p)
                layers.append(dropout_layer)

        self.features = nn.Sequential(*layers)
        num_flat_features = in_channels * h * w
        self.classifier = nn.Linear(num_flat_features, num_classes)
        self.model = nn.Sequential(self.features, nn.Flatten(), self.classifier)

        self.criterion = nn.MSELoss() # changed loss function and got error!
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.save_hyperparameters()

        self.apply(self._init_weights)  # Apply weight initialization

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        - Conv2d and Linear layers: Kaiming initialization
        - BatchNorm layers: set weight to 1 and bias to 0
        """
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        #print('is data on GPU??')
        #print(inputs.device)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)
        self.train_acc = self.accuracy(outputs.argmax(dim=1), labels)

        # Log the training loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # Access the training dataloader
        data_loader = self.trainer.datamodule.train_dataloader()
        num_classes = self.hparams.num_classes
        dataset_size = len(data_loader.dataset)
        depth = len([layer for layer in self.model.modules() if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))])

        # Compute the generalization bound using the utility function
        bound = our_total_bound(self, data_loader, num_classes, dataset_size, depth)
        self.log("generalization_bound", bound, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)
        self.val_acc = self.accuracy(outputs.argmax(dim=1), labels)
        
        # Log the validation loss and accuracy
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)  
        self.test_acc = self.accuracy(outputs.argmax(dim=1), labels)

        # Log the test loss and accuracy
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

        # Access the training dataloader
        data_loader = self.trainer.datamodule.train_dataloader()
        num_classes = self.hparams.num_classes
        dataset_size = len(data_loader.dataset)
        depth = len([layer for layer in self.model.modules() if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))])

        # Compute the generalization bound using the utility function
        bound = our_total_bound(self, data_loader, num_classes, dataset_size, depth)
        self.log("generalization_bound", bound, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Define the scheduler with cosine decay and warm-up
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return current_step / self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            return cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Adjust every step
                "frequency": 1       # Apply every step
            }
        }
    



def simulate_model_dimensions(kernel_dict, max_pool_layer_dict, dropout_layer_dict, input_dims, dataset_name="MNIST"):
    """
    Simulates the dimensions of the data as it passes through the model layers.
    
    Args:
        kernel_dict (dict): Nested dictionary describing the convolutional layers.
        max_pool_layer_dict (dict): Dictionary describing the max pooling layers.
        dropout_layer_dict (dict): Dictionary describing the dropout layers.
        input_dims (tuple): A tuple (height, width, channels) representing the input dimensions.
        dataset_name (str): Name of the dataset (e.g., "MNIST", "CIFAR10").
    """
    # Adjust input channels based on dataset
    if dataset_name == "CIFAR10":
        in_channels = 3
        h, w = 32, 32  # CIFAR-10 images are 32x32
    elif dataset_name == "MNIST":
        in_channels = 1
        h, w = 28, 28  # MNIST images are 28x28
    else:
        h, w, in_channels = input_dims  # Default dimensions if provided explicitly

    print(f"Dataset: {dataset_name}")
    print(f"Input Dimensions: Height={h}, Width={w}, Channels={in_channels}\n")
    
    num_layers = max(
        max(kernel_dict.keys(), default=0),
        max(max_pool_layer_dict.keys(), default=0),
        max(dropout_layer_dict.keys(), default=0)
    )
    
    for layer_idx in range(num_layers + 1):
        print(f"Layer {layer_idx}:")
        
        # Convolutional Layer
        if layer_idx in kernel_dict:
            layer_params = kernel_dict[layer_idx]
            kernel_size = layer_params['kernel_size']
            out_channels = layer_params['out_channels']
            stride = layer_params.get('stride', 1)
            padding = layer_params.get('padding', 0)
            
            # Compute output dimensions after convolution
            h_out = (h + 2 * padding - kernel_size) // stride + 1
            w_out = (w + 2 * padding - kernel_size) // stride + 1
            
            print(f"  Conv2d: kernel_size={kernel_size}, stride={stride}, padding={padding}")
            print(f"    Input Channels: {in_channels}")
            print(f"    Output Channels: {out_channels}")
            print(f"    Output Dimensions: Height={h_out}, Width={w_out}")
            
            # Update dimensions and channels for next layer
            h, w = h_out, w_out
            in_channels = out_channels
            
        # Max Pooling Layer
        if layer_idx in max_pool_layer_dict:
            layer_params = max_pool_layer_dict[layer_idx]
            pool_size = layer_params['pool_size']
            stride = layer_params.get('stride', pool_size)
            padding = layer_params.get('padding', 0)
            
            # Compute output dimensions after pooling
            h_out = (h + 2 * padding - pool_size) // stride + 1
            w_out = (w + 2 * padding - pool_size) // stride + 1
            
            print(f"  MaxPool2d: pool_size={pool_size}, stride={stride}, padding={padding}")
            print(f"    Output Dimensions: Height={h_out}, Width={w_out}")
            
            # Update dimensions for next layer
            h, w = h_out, w_out
            
        # Dropout Layer
        if layer_idx in dropout_layer_dict:
            p = dropout_layer_dict[layer_idx]
            print(f"  Dropout: p={p}")
            # Dropout does not change dimensions
            
        print("")  # Empty line for readability
    
    # After all layers
    num_flat_features = in_channels * h * w
    print(f"Final Output Dimensions before Linear Layer: {num_flat_features}")
    
    if num_flat_features <= 0:
        print("Error: The number of features before the linear layer is non-positive.")
        print("Possible causes:")
        print("  - The dimensions have been reduced too much due to convolution or pooling.")
        print("  - Incorrect padding, stride, or kernel sizes leading to zero or negative dimensions.")
    else:
        print("The dimensions are valid for the linear layer input.")

    return num_flat_features