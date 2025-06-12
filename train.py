"""
After extracting the dataset and running the split.py script,
this script orchestrates the training and pruning of the models.
"""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

from DataLoading import get_train_data_pipeline, get_validation_data_pipeline
from Models import ShuffleNetV2X05, MobileNetV2, MobileNetV3, ResNet18

BATCH_SIZE = 4
WORKERS = 8

# augment the data by random resizing, rotation and lighting changes
train_transforms = get_train_data_pipeline()
val_transforms = get_validation_data_pipeline()

training_data = ImageFolder("split_train", train_transforms)
validation_data = ImageFolder("split_val", val_transforms)

dataloaders = {"train": DataLoader(training_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=WORKERS), "val": DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=WORKERS)}

num_classes = len(training_data.classes)

# Uncomment for Resnet18
rnet18= ResNet18(num_classes)
# Uncomment for Mobilenet v2/v3
mnetv2 = MobileNetV2(num_classes)
mnetv3 = MobileNetV3(num_classes)
# Uncomment for ShuffleNet
snetv2 = ShuffleNetV2X05(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer_ft = lambda model_ft: optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lambda optimizer_ft: lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

mnetv2.with_environment(dataloaders["train"], dataloaders["val"], criterion, optimizer_ft, exp_lr_scheduler).train_and_prune()
mnetv3.with_environment(dataloaders["train"], dataloaders["val"], criterion, optimizer_ft, exp_lr_scheduler).train_and_prune()
snetv2.with_environment(dataloaders["train"], dataloaders["val"], criterion, optimizer_ft, exp_lr_scheduler).train_and_prune()
rnet18.with_environment(dataloaders["train"], dataloaders["val"], criterion, optimizer_ft, exp_lr_scheduler).train_and_prune()
