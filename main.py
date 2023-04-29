# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools
 
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


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
    args = parse()
    device = get_default_device()
    print(device)

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")
    merged_df = pd.merge(x, y[['patientunitstayid', 'hospitaldischargestatus']], on='patientunitstayid')
    processed_x_train = preprocess_x(merged_df)
    # processed_x_test = preprocess_x(y)

    # train_x, train_y, test_x, test_y = split_data(x, y)

    # ###### Your Code Here #######
    # # Add anything you want here

    # ############################

    # ###### Your Code Here #######
    # # Add anything you want here

    # ############################

    # model = Model(args)  # you can add arguments as needed
    # model.fit(processed_x_train, train_y)
    # x = load_data("test_x.csv")

    # ###### Your Code Here #######
    # # Add anything you want here

    # ############################

    # processed_x_test = preprocess_x(x)

    # prediction_probs = model.predict_proba(processed_x_test)

    #### Your Code Here ####
    # Save your results

    ########################


if __name__ == "__main__":
    main()
