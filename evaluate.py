"""
After running the train.py script, the results can
be evaluated using this script.
"""

import os
import csv
import torch
from paretoset import paretoset
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

from ptflops import get_model_complexity_info

from DataLoading import get_train_data_pipeline, get_validation_data_pipeline

import matplotlib.pyplot as plt

def compile_training_losses(prefix: str, sorted_folder_paths: list[str]):
    # compiles training and validation loss/accuracy for a model
    # folder paths is assumed to be in the correct order
    val_losses = []
    train_losses = []
    val_accuracy = []
    train_accuracy = []

    for path in sorted_folder_paths:
        complete_path = f"{prefix}/{path}/losses.csv"
        if not os.path.isfile(complete_path):
            continue
        
        f = open(complete_path, mode="r")
        all_losses = list(csv.DictReader(f))

        for loss_entry in all_losses:
            if loss_entry["Phase"] == "train":
                train_losses.append(float(loss_entry["Loss"]))
                train_accuracy.append(float(loss_entry["Accuracy"]))
            else:
                val_losses.append(float(loss_entry["Loss"]))
                val_accuracy.append(float(loss_entry["Accuracy"]))
    return train_losses, train_accuracy, val_losses, val_accuracy

def output_loss_graph(data, ylabel, xlabel, fname):
    fig, ax = plt.subplots()
    fig.suptitle(f"{xlabel} vs {ylabel}")
    ax.plot(data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(fname)

def evaluate_models(prefix_path, sorted_folder_paths_and_names, datasets):
    """
    Evaluate a sequence of incrementally pruned models
    on train/val/test data, and record their parameter
    count and MMac values
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # output format: Model, Ratio, TrainAcc, ValAcc, TestAcc, ParameterCount, MMac
    results = []

    example_instance = next(iter(datasets[next(iter(datasets.keys()))]))[0]
    
    for path, name in sorted_folder_paths_and_names:
        full_path = f"{prefix_path}/{path}/model.pth"

        if not os.path.isfile(full_path):
            continue

        torch.cuda.empty_cache() # we go out of memory if we dont do this
        model = torch.load(full_path, weights_only = False).to(device)

        macs, params = get_model_complexity_info(model, tuple(example_instance.shape[1:]), as_strings = False, print_per_layer_stat = False, verbose = False)

        result = {
            "Model": prefix_path.split("/")[1],
            "Ratio": path,
            "ParameterCount": float(params),
            "FLOPS": float(macs)
        }

        for dataset in datasets:
            running_corrects = 0
            running_misses = 0
            done = 0
            for inputs, labels in datasets[dataset]:
                print(f"{int(done*100/len(datasets[dataset]))}%", end="\r")
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels)
                running_misses += torch.sum(preds != labels)
                done +=1
            result[dataset] = float(running_corrects / (running_misses + running_corrects))
        results.append(result)
    return results
    
if __name__ == "__main__":

    # Load the datasets
    train_transforms = get_train_data_pipeline()
    val_transforms = get_validation_data_pipeline()
    
    training_data = ImageFolder("split_train", train_transforms)
    validation_data = ImageFolder("split_val", val_transforms)
    test_data = ImageFolder("split_test", val_transforms)

    dataloaders = {
        "train": DataLoader(training_data, batch_size = 32, num_workers=4),
        "val": DataLoader(validation_data, batch_size = 32, num_workers=4),
        "test": DataLoader(test_data, batch_size = 32, num_workers=4)
    }
    
    top_folder = "training_results"
    
    # what models are there?
    model_folder_paths = next(os.walk(top_folder))[1]

    models = [] # holds dicts: ModelName, Ratio, TrainAcc, ValAcc, TestAcc, ParameterCount, MMac
    
    for meta_model in model_folder_paths:
        full_path = f"{top_folder}/{meta_model}"
        folder_names = next(os.walk(full_path))[1]
        ratios = [float(r) for r in folder_names]

        _l = list(zip(ratios, folder_names))
        _l.sort(key=lambda a: a[0])
        sorted_folder_paths = [a[1] for a in _l]

        # Step 1 : Visualize Training Process
        train_losses, train_accuracy, val_losses, val_accuracy = compile_training_losses(full_path, sorted_folder_paths)

        output_loss_graph(train_losses, "Training Loss", "Time", f"{meta_model}_train_loss")
        output_loss_graph(train_accuracy, "Training Accuracy", "Time", f"{meta_model}_train_acc")
        output_loss_graph(val_losses, "Validation Loss", "Time", f"{meta_model}_val_loss")
        output_loss_graph(val_accuracy, "Validation Accuracy", "Time", f"{meta_model}_val_acc")
        
        # Step 2: Load Model, Evaluate on Train, Validation, Test set
        results = evaluate_models(full_path, zip(sorted_folder_paths, ratios), dataloaders)
        
        f = open(f"{meta_model}_pruning_results.csv", mode="w")
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

        models.extend(results)

    # Find the pareto optimal (in terms of FLOPS, Parameters, Test Accuracy) models
    result_pd = pd.DataFrame.from_dict(models)
    mask = paretoset(result_pd[["test", "ParameterCount", "FLOPS"]], sense = ["max", "min", "min"])
    efficient_results = result_pd[mask]

    print(efficient_results)
    # save the pareto optimal model performances
    efficient_results.to_csv("pareto_efficient_pruned_models.csv", index=False)
