import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import yaml
with open("./CONFIG.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class CatsDogsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform['train'])
            train_size = int((1 - self.val_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform['val'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class CatsDogsModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(CatsDogsModel, self).__init__()
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

def train(data_dir):

    data_module = CatsDogsDataModule(data_dir)
    model = CatsDogsModel(num_classes=2)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='my_checkpoints',
        filename='cats-dogs-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    trainer = Trainer(max_epochs=25, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)

if __name__ == "__main__":

    train(cfg["DATA_DIR"])
