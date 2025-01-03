import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import Accuracy
import lightning as L
from lightning import LightningModule
import math
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init
import math
import os
import csv


from utils import max_pixel_sums, eval_rho, our_total_bound

def effective_rank(tensor, epsilon=1e-12):
    try:
        # Reshape tensor to 2D
        W = tensor.view(tensor.size(0), -1)

        # Normalize to improve numerical stability
        W = W / W.norm(p='fro')

        # Compute SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Filter small singular values
        S = S[S > epsilon]
        if S.numel() == 0:
            return 0.0

        # Compute entropy and effective rank
        p = S / S.sum()
        H = -torch.sum(p * torch.log(p))
        eff_rank = torch.exp(H)
        return eff_rank

    except torch._C._LinAlgError as e:
        print("SVD failed:", e)
        return 0.0  # Fallback rank

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
        bound, mult1, mult2, mult3, add1 = our_total_bound(self, data_loader, num_classes, dataset_size, depth)

        # Log the bound as before
        self.log("train_generalization_bound", bound, prog_bar=True)

        # Log the intermediate values to a "folder" named "bound_components" in W&B
        self.log("bound_components/train_mult1", mult1, prog_bar=False)
        self.log("bound_components/train_mult2", mult2, prog_bar=False)
        self.log("bound_components/train_mult3", mult3, prog_bar=False)
        self.log("bound_components/train_add1", add1, prog_bar=False)

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
        bound, mult1, mult2, mult3, add1 = our_total_bound(self, data_loader, num_classes, dataset_size, depth)

        # Log the bound as before
        self.log("test_generalization_bound", bound, prog_bar=True)

        # Log the intermediate values to a "folder" named "bound_components" in W&B
        self.log("bound_components/test_mult1", mult1, prog_bar=False)
        self.log("bound_components/test_mult2", mult2, prog_bar=False)
        self.log("bound_components/test_mult3", mult3, prog_bar=False)
        self.log("bound_components/test_add1", add1, prog_bar=False)

        return loss

    def configure_optimizers(self):
        # Define SGD optimizer with weight decay
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class ModularCNN(LightningModule):
    def __init__(self, kernel_dict, max_pool_layer_dict, dropout_layer_dict, num_classes, 
                 weight_decay, 
                 lr, optimizer_choice, momentum,  
                 use_warmup, lr_decay_type, warmup_steps, max_steps,
                 weight_init=True):
        super(ModularCNN, self).__init__()
        # implement model with N number of Conv2d layers followed by potential maxpooling layers, relu activations, dropout layers, and one single final linear layer
        # input size is always 28x28
        # kernel_dict is a nested dictinary that describes the kernel size for each conv layer including the number of output channels for each conv layer. 1st keys are the layer indices. 2nd keys include the following: kernel_size, output_channels, stride, padding
        # max_pool_layer_dict describes the max pool size for each max pool layer and at which layer it is applied
        # dropout_layer_dict describes the dropout rate for each dropout layer and at which layer it is applied

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_choice = optimizer_choice
        self.momentum = momentum
        
        self.use_warmup = use_warmup

        self.lr_decay_type = lr_decay_type
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self.kernel_dict = kernel_dict
        self.num_classes = num_classes
        self.max_pixel_sum = None

        self.weight_init = weight_init

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
                conv_layer = weight_norm(conv_layer, dim=None)
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

        self.weight_layers = []
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self.weight_layers.append(layer)

        self.csv_file_path = os.path.join("csv_files", "layer_stats.csv")
        self.csv_header_written = False

        if self.weight_init:
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
       
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)
        self.train_acc = self.accuracy(outputs.argmax(dim=1), labels)

        # Log the training loss and accuracy
        #self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        data_loader = self.trainer.datamodule.train_dataloader()
        num_classes = self.hparams.num_classes
        dataset_size = len(data_loader.dataset)
        depth = len([layer for layer in self.model.modules() if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))])

        # Compute the generalization bound using the utility function
        bound, mult1, mult2, mult3, add1 = our_total_bound(self, num_classes, dataset_size, depth, self.kernel_dict, self.max_pixel_sum)

        self.log("generalization_bound", bound, on_epoch=False, on_step=True, prog_bar=False)
        self.log("bound_components/mult1", mult1, on_epoch=False, on_step=True, prog_bar=False)
        self.log("bound_components/mult2", mult2, on_epoch=False, on_step=True, prog_bar=False)
        self.log("bound_components/mult3", mult3, on_epoch=False, on_step=True, prog_bar=False)
        self.log("bound_components/add1", add1, on_epoch=False, on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)
        self.val_acc = self.accuracy(outputs.argmax(dim=1), labels)
        
        # Log the validation loss and accuracy
        #self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.hparams.num_classes).float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels_one_hot)  
        self.test_acc = self.accuracy(outputs.argmax(dim=1), labels)

        # Log the test loss and accuracy
        #self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

        # Access the training dataloader
        data_loader = self.trainer.datamodule.train_dataloader()
        num_classes = self.hparams.num_classes
        dataset_size = len(data_loader.dataset)
        depth = len([layer for layer in self.model.modules() if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear))])

        # Compute the generalization bound using the utility function
        bound, mult1, mult2, mult3, add1 = our_total_bound(self, num_classes, dataset_size, depth, self.kernel_dict, self.max_pixel_sum)

        # Log the bound as before
        self.log("generalization_bound", bound, prog_bar=True)

        # Log the intermediate values to a "folder" named "bound_components" in W&B
        self.log("bound_components/mult1", mult1, prog_bar=False)
        self.log("bound_components/mult2", mult2, prog_bar=False)
        self.log("bound_components/mult3", mult3, prog_bar=False)
        self.log("bound_components/add1", add1, prog_bar=False)

        return loss

    def configure_optimizers(self):
        # Choose optimizer based on self.optimizer_choice
        if self.optimizer_choice == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_choice == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError("Unsupported optimizer choice. Use 'SGD' or 'AdamW'.")

        # Define lr_lambda function combining warmup and decay
        if self.lr_decay_type == 'multi_step':
            scheduler = MultiStepLR(
                optimizer,
                milestones=[60, 100, 300],
                gamma=0.2
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",  # 'epoch' or 'step' based on your needs
                "frequency": 1,
                "name": "MultiStepLR"
            }
        else:
            # Handle other lr_decay_types if necessary
            def lr_lambda(current_step):
                if self.use_warmup and current_step < self.warmup_steps:
                    return current_step / self.warmup_steps
                else:
                    progress = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                    if self.lr_decay_type == 'cosine':
                        return 0.5 * (1 + math.cos(math.pi * progress))
                    elif self.lr_decay_type == 'linear':
                        return max(0.0, 1 - progress)
                    elif self.lr_decay_type == 'no_decay':
                        return 1.0  # No decay
                    else:
                        raise ValueError("Unsupported lr decay type. Use 'cosine', 'linear', 'no_decay', or 'multi_step'.")

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",  # 'epoch' or 'step' based on your needs
                "frequency": 1
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }
    
    '''
    def on_fit_start(self):
        # Create the directory if it doesn't exist
        os.makedirs("csv_files", exist_ok=True)

        # Prepare CSV header: one column for step, then norm_i and rank_i for each layer
        header = ["step"]
        for i, _ in enumerate(self.weight_layers):
            header.append(f"norm_{i}")
            header.append(f"rank_{i}")

        # Write the header
        with open(self.csv_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        self.csv_header_written = True
    '''
    
    def on_train_epoch_end(self):
        # Ensure self.trainer and self.hparams are accessible
        if self.trainer.datamodule and self.hparams:
            if self.current_epoch % 5 == 0:
                # Iterate over all Conv2d/Linear layers and log their Frobenius norms and ranks
                layer_idx = 0
                for layer in self.model.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        w = layer.weight
                        # Frobenius norm of the weight matrix
                        fro_norm = torch.norm(w, p='fro')

                        # Compute rank: reshape weight to a 2D matrix and compute matrix rank
                        w_matrix = w.view(w.size(0), -1)
                        rank = effective_rank(w_matrix)

                        # Log these values
                        # Using a consistent naming scheme groups them in W&B:
                        # "weight_fro_norm/layer_0", "weight_fro_norm/layer_1", etc.
                        self.log(f"weight_fro_norm/layer_{layer_idx}", fro_norm, on_step=False, on_epoch=True)
                        self.log(f"weight_ranks/layer_{layer_idx}", rank, on_step=False, on_epoch=True)
                        layer_idx += 1
        '''

            # Compute Frobenius norm and effective rank for each layer
            norms = []
            ranks = []
            for layer in self.weight_layers:
                w = layer.weight
                fro_norm = torch.norm(w, p='fro')
                eff_r = effective_rank(w)
                norms.append(fro_norm.item())
                ranks.append(eff_r.item())
        
            # Append a row to the CSV file
            step = self.global_step
            row = [step]
            for n, r in zip(norms, ranks):
                row.append(n)
                row.append(r)

            with open(self.csv_file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        '''