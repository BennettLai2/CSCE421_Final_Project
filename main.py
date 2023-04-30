# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools
 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score


from data import load_data, preprocess_x, split_data
from parser_1 import parse
from model import Model, MortalityDataset, train_model
from torch.utils.data import DataLoader, Dataset


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
    args = parse()
    device = get_default_device()
    print(device)

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")
    merged_df = pd.merge(x, y[['patientunitstayid', 'hospitaldischargestatus']], on='patientunitstayid')
    processed_x_train = preprocess_x(merged_df)
    # print(processed_x_train.dtypes)
    # processed_x_test = preprocess_x(y)

    # train_x, train_y, test_x, test_y = split_data(x, y)

    train_dataset = MortalityDataset(processed_x_train, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    input_size = processed_x_train.shape[1]
    hidden_size = 64
    num_layers = 3
    output_size = 1
    model = Model(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")


if __name__ == "__main__":
    main()
