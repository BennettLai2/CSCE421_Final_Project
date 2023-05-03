# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools
import numpy as np

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data import load_data, preprocess_x, split_data
from parser_1 import parse
from model import Model


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def main():
    # x = load_data("train_x.csv")
    # x = preprocess_x(x)
    # x.to_csv('processed_train_x.csv')
    x = load_data("processed_train_x.csv")
    y = load_data('train_y.csv')

    y = pd.DataFrame({'hospitaldischargestatus': x['patientunitstayid'].map(
        y.set_index('patientunitstayid')['hospitaldischargestatus'])})
    device = get_default_device()

    train_x, test_x, train_y, test_y = split_data(x, y)

    # -------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=False)

    input_size = train_x.shape[1] - 1  # Minus 1 to exclude 'patientunitstayid'
    hidden_size = 1024
    num_layers = 2
    output_size = 1

    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

        cv_model = Model(input_size, hidden_size, num_layers, output_size, device)  # you can add arguments as needed
        acc = cv_model.fit(X_train, Y_train, X_test, Y_test)
        print(acc)

    model = Model(input_size, hidden_size, num_layers, output_size, device)  # you can add arguments as needed
    acc = model.fit(train_x, train_y, test_x, test_y)
    print(acc)


if __name__ == "__main__":
    main()

