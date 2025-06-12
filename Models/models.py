import os
import copy
import time
import csv
from typing import Callable

import torch
import torch_pruning as tp
import torch.nn as nn
from ptflops import get_model_complexity_info
from torchvision import models
from tempfile import TemporaryDirectory

class BaseClassificationModel:
    number_of_classes: int
    name: str = "BaseClassificationModel"
    store_prefix_path: str = "training_results"
    
    def __init__(self, num_clases: int):
        """
        Base classfication model interface for training and inference.

        Parameters:
          num_classes: number of classes in Classification problem
        """
        raise RuntimeError("Unimplemented Base Constructor")


    def get_last_layer(self):
        """
        Get the last layer of the model.
        """
        pass
    
    def with_environment(self, train_dataloader, validation_dataloader, criterion, optimizer_maker, scheduler_maker):
        """
        Specifcy training environment.

        Parameters:
          train_dataloader: torch dataloader for the training set
          validation_dataloader: torch dataloader for the validation set
          criterion: torch loss function (such as nn.CrossEntropyLoss())
          optimizer_maker: function, that when called with the torch model yields a optimizer object
          scheduler_maker: function, that when called with the torch optimizer yields a  torch scheduler (such as StepLR())
        """
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.criterion = criterion
        self.optimizer_maker = optimizer_maker
        self.scheduler_maker = scheduler_maker
        return self

    def train_and_prune(self, initial_ratio: float = 0.05, target_ratio: float = 1.0, steps = 10, epochs_per_step = 10):
        """
        Run Automated Gradual Pruning.

        Starting from `initial_ratio`, run `steps` pruning epochs resulting in
        a final pruning ratio of `target_ratio`.
        Each pruning step has `epochs_per_step` training epochs inbetween.

        Parameters:
          initial_ratio: initial pruning ratio for the first step
          target_ratio: target pruning ratio to arrive at
          steps: how many pruning steps
          epochs_per_step: how many training epochs between the steps

        Returns:
          `best_models` the saved file names of the best models (in terms
            of validation error) of each pruning step
        """
        if not self.train_dataloader or not self.validation_dataloader or not self.criterion or not self.optimizer_maker or not self.scheduler_maker:
            raise RuntimeError("call to .train() without training environment (try calling .with_environment()) before")

        # output folders setup
        model_output_path = f"{self.store_prefix_path}/{self.name}"
        if not os.path.isdir(self.store_prefix_path):
            os.mkdir(self.store_prefix_path)
        if not os.path.isdir(model_output_path):
            os.mkdir(model_output_path)
        
        # pruning algorithm
        
        ratios = [0.0] # start with zero pruning
        ratios.extend(list(torch.linspace(initial_ratio, target_ratio, steps=steps)))

        train_instance, _ = next(iter(self.train_dataloader))
        example_inputs = torch.randn(train_instance.shape)

        importance = tp.importance.MagnitudeImportance(p=1)
        
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
        for step, ratio in enumerate(ratios):
            print(f"AGP {self.name} Pruning Step {step} of {steps}, Ratio {ratio}")

            pruner = tp.pruner.MagnitudePruner(
                model=self.model.to(device),
                example_inputs=example_inputs.to(device),
                importance = importance,
                pruning_ratio = float(ratio),
                ignored_layers = [self.get_last_layer()]
            )
            
            pruner.step()

            # clone the model here, because ptflops injects methods into the model (that causes issues with torch.save/load mechanism)
            macs, params = get_model_complexity_info(copy.deepcopy(self.model), tuple(train_instance.shape[1:]), as_strings = True, print_per_layer_stat = False, verbose = False)
            print(f"Flops/Params {macs}/{params}")

            # prune ratio best model folder
            ratio_path = f"{model_output_path}/{ratio}"
            if not os.path.isdir(ratio_path):
                os.mkdir(ratio_path)
            self.train(epochs_per_step, best_model_path = ratio_path)
            


        print(f"AGP {self.name} Pruning Step {steps} of {steps}, Ratio {target_ratio}")
        ratio_path = f"{model_output_path}/{target_ratio}"
        if not os.path.isdir(ratio_path):
            os.mkdir(ratio_path)
        self.train(epochs_per_step, best_model_path = ratio_path)
    
    def train(self, num_epochs=25, best_model_path="default"):
        """
        Train the model.

        Uses the parameters from the previous 'model.with(...)' call.

        Parameters:
          num_epochs: number of epochs to train for
          best_model_folder: the subfolder where the parameters of the
           best model will be saved

        """

        if not self.train_dataloader or not self.validation_dataloader or not self.criterion or not self.optimizer_maker or not self.scheduler_maker:
            raise RuntimeError("call to .train() without training environment (try calling .with_environment()) before")

        best_model_params_path = f"{best_model_path}/model.pth"
        epoch_performance_path = f"{best_model_path}/losses.csv"

        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
        self.model.to(device)
        
        since = time.time()

        optimizer = self.optimizer_maker(self.model)
        scheduler = self.scheduler_maker(optimizer)

        dataloaders = {
            "train": self.train_dataloader,
            "val": self.validation_dataloader
        }

        data_sizes = {
            "train": len(self.train_dataloader.dataset),
            "val": len(self.validation_dataloader.dataset)
        }
        
        with TemporaryDirectory() as tempdir:
            best_acc = 0.0

            metrics = []

            for epoch in range(num_epochs):
                print(f"{self.name} Epoch {epoch}/{num_epochs-1}")
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
                            loss = self.criterion(outputs, labels)

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

                    metrics.append({
                        "Index": epoch,
                        "Phase": phase,
                        "Loss": float(epoch_loss),
                        "Accuracy": float(epoch_acc)
                    })

                    print(f"{phase} loss: {epoch_loss} Acc: {epoch_acc}")

                    if phase == "val" and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model, best_model_params_path)
                print()

            time_elapsed = time.time() - since
            print(f"took {time_elapsed / 60}, best {best_acc}")

            f = open(epoch_performance_path, mode="w")
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
            
            return self

class ShuffleNetV2X05(BaseClassificationModel):
    name: str = "ShuffleNet_v2_x0_5"
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.shufflenet_v2_x0_5(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    def get_last_layer(self):
        return self.model.fc

class MobileNetV2(BaseClassificationModel):
    name: str = "MobileNet_v2"
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    def get_last_layer(self):
        return self.model.classifier[-1]

class MobileNetV3(BaseClassificationModel):
    name: str = "MobileNet_v3_small"
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    def get_last_layer(self):
        return self.model.classifier[-1]

class ResNet18(BaseClassificationModel):
    name: str = "ResNet18"
    def __init__(self, num_classes):
        self.number_of_classes = num_classes
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    def get_last_layer(self):
        return self.model.fc
