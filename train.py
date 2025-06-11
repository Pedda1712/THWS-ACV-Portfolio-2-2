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

from Models import ShuffleNetV2X05, MobileNetV2, MobileNetV3, ResNet18

BATCH_SIZE = 4
WORKERS = 4

# augment the data by random resizing, rotation and lighting changes
train_transforms = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomRotation(20),
    v2.ColorJitter(brightness = (0.3, 1.7), contrast = (0.5, 1.5)),
    v2.RandomPerspective(),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = v2.Compose([
    v2.Resize(256), # let's hope this does not lead to problems with aspect ratio
    v2.CenterCrop(224),
    
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

training_data = ImageFolder("split_train", train_transforms)
validation_data = ImageFolder("split_val", val_transforms)

dataloaders = {"train": DataLoader(training_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=WORKERS), "val": DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=WORKERS)}

num_classes = len(training_data.classes)

# Uncomment for Resnet18
model = ResNet18(num_classes)
# Uncomment for Mobilenet v2/v3
#model = MobileNetV2(num_classes)
#model = MobileNetV3(num_classes)
# Uncomment for ShuffleNet
#model = ShuffleNetV2X05(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer_ft = lambda model_ft: optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lambda optimizer_ft: lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model.train(dataloaders["train"], dataloaders["val"], criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 2)
