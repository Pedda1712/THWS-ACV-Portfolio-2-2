"""
Split the dataset into Train/Validation/Test components.

Note: Train/Test comes pre-split, so we only need
      to extract a validation set from the train set.

Here we need to take care to extract entire 'sequences'
of the same traffic sign in the same context, so that
no data leakage occurs into the validation set.
"""

import pandas as pd
import os, shutil

VALIDATION_SIZE = 0.2

if __name__ == "__main__":
    csv = pd.read_csv("Data/Train.csv")[["ClassId", "Path"]]
    test_csv = pd.read_csv("Data/Test.csv")[["ClassId", "Path"]]

    test_data = dict([(c, []) for c in test_csv["ClassId"].unique()])
    for row in test_csv.itertuples():
        c = row.ClassId
        p = row.Path
        test_data[c].append(p)

    data = dict([(c, {}) for c in csv["ClassId"].unique()])
    # sort images into class-series structure
    for row in csv.itertuples():
        c = row.ClassId
        p = row.Path
        series_id = int(p.split('_')[1])
        if not series_id in data[c]:
            data[c][series_id] = [p]
        else:
            data[c][series_id].append(p)

    # create target directories
    if os.path.isdir("split_train") or os.path.isdir("split_val") or os.path.isdir("split_test"):
        print("target directories already exist, delete or move the old splits first!")
        exit()

    os.mkdir("split_train")
    os.mkdir("split_val")
    os.mkdir("split_test")

    # copy the train/val data into the respective directories
    for _class in data:
        os.mkdir(f"split_train/{_class}")
        os.mkdir(f"split_val/{_class}")

        series = list(data[_class].values())
        split_point = int(len(series) * VALIDATION_SIZE)
        
        validation_paths = [x for xs in series[:split_point] for x in xs]
        train_paths = [x for xs in series[split_point:] for x in xs]

        for p in validation_paths:
            shutil.copyfile("Data/" + p, f"split_val/{_class}/{p.split('/')[-1]}")

        for p in train_paths:
            shutil.copyfile("Data/" + p, f"split_train/{_class}/{p.split('/')[-1]}")


    # create test data directory structure
    for _class in test_data:
        os.mkdir(f"split_test/{_class}")
        for p in test_data[_class]:
            shutil.copyfile("Data/" + p, f"split_test/{_class}/{p.split('/')[-1]}")
