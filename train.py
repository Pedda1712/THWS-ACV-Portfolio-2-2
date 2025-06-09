import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

BATCH_SIZE = 4
WORKERS = 4

# augment the data by random resizing, rotation and lighting changes
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

training_data = ImageFolder("split_train", train_transforms)
validation_data = ImageFolder("split_val", val_transforms)

data_sizes = {
    "train": len(training_data),
    "val": len(validation_data)
}

dataloaders = {"train": DataLoader(training_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=WORKERS), "val": DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle=True, num_workers=WORKERS)}

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs-1}")
            print("=" * 15)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0
                done = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if done % 1 == 0:
                        print(f"{(100*BATCH_SIZE*done/data_sizes[phase])}%", end="\r")

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
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
                    torch.save(model.state_dict(), best_model_params_path)
            print()


        time_elapsed = time.time() - since

        print(f"took {time_elapsed / 60}, best {best_acc}")
        model.load_state_dict(torch.load(best_model_params_path, weights_only = True))
    return model

model_ft = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(training_data.classes))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
