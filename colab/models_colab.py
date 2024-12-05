import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import Accuracy
import pytorch_lightning as L
from pytorch_lightning import LightningModule
import math
import numpy as np

from utils_colab import max_pixel_sums, eval_rho, our_total_bound


class SparseDeepModel(LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        super(SparseDeepModel, self).__init__()

        if model_name == 'InceptionV3':
            self.model = models.inception_v3(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'AlexNet':
            self.model = models.alexnet(pretrained=False)
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
        # Define Adam optimizer with weight decay
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
