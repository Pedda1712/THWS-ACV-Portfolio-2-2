import torch
from torchvision.transforms import v2

def get_train_data_pipeline():
    return v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness = (0.3, 1.7), contrast = (0.5, 1.5)),
        v2.RandomPerspective(),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_validation_data_pipeline():
    return v2.Compose([
        v2.Resize(256), 
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
