import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchmetrics import Accuracy
import lightning as L
from lightning import LightningModule







class SparseDeepModel(LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay):
        super(SparseDeepModel, self).__init__()

        if model_name == 'InceptionV3':
            self.model = models.inception_v3(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'AlexNet':
            self.model = models.alexnet(pretrained=False)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)


        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.train_acc = self.accuracy(outputs, labels)

        # Log the training loss and accuracy
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.val_acc = self.accuracy(outputs, labels)
        
        # Log the validation loss and accuracy
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.test_acc = self.accuracy(outputs, labels)

        # Log the test loss and accuracy
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Define Adam optimizer with weight decay
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    