import os
import time
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models
from tempfile import TemporaryDirectory

class BaseClassificationModel:
    number_of_classes: int
    
    def __init__(self, num_clases: int):
        """
        Base classfication model interface for training and inference.

        Parameters:
          num_classes: number of classes in Classification problem
        """
        raise RuntimeError("Unimplemented Base Constructor")

    def train(self, train_dataloader, validation_dataloader, criterion, optimizer_maker, scheduler_maker, num_epochs=25):
        """
        Train the model.

        Parameters:
          train_dataloader: torch dataloader for the training set
          validation_dataloader: torch dataloader for the validation set
          criterion: torch loss function (such as nn.CrossEntropyLoss())
          optimizer_maker: function, that when called with the torch model yields a optimizer object
          scheduler_maker: function, that when called with the torch optimizer yields a  torch scheduler (such as StepLR())
        """
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        since = time.time()

        optimizer = optimizer_maker(self.model)
        scheduler = scheduler_maker(optimizer)

        dataloaders = {
            "train": train_dataloader,
            "val": validation_dataloader
        }

        data_sizes = {
            "train": len(train_dataloader.dataset),
            "val": len(validation_dataloader.dataset)
        }
        
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f"Epoch {epoch}/{num_epochs-1}")
                print("=" * 15)

                for phase in ["train", "val"]:
                    if phase == "train":
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0.0
                    done = 0

                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        if done % 100 == 0:
                            print(f"{int(100*done/len(dataloaders[phase]))}%", end="\r")

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                        done += 1
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / data_sizes[phase]
                    epoch_acc = running_corrects.double() / data_sizes[phase]

                    print(f"{phase} loss: {epoch_loss} Acc: {epoch_acc}")

                    if phase == "val" and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model.state_dict(), best_model_params_path)
                print()

            time_elapsed = time.time() - since
            print(f"took {time_elapsed / 60}, best {best_acc}")
            self.model.load_state_dict(torch.load(best_model_params_path, weights_only = True))
            return self

class ShuffleNetV2X05(BaseClassificationModel):
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.shufflenet_v2_x0_5(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

class MobileNetV2(BaseClassificationModel):
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

class MobileNetV3(BaseClassificationModel):
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

class ResNet18(BaseClassificationModel):
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

