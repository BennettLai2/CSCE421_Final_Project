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
    # args = parse()
    # device = get_default_device()
    # print(device)

    merged_df = load_data("processed_train_x.csv")
    # x = load_data("train_x.csv")
    # y = load_data("train_y.csv")
    # merged_df = pd.merge(x, y[['patientunitstayid', 'hospitaldischargestatus']], on='patientunitstayid')
    # merged_df = preprocess_x(merged_df)
    # merged_df = merged_df.reindex(columns =['patientunitstayid', 'hospitaldischargestatus', 'ethnicity_African American', 'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic', 'ethnicity_Native American', 'ethnicity_Other/Unknown', 'gender_Female', 'gender_Male', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'Capillary Refill', 'GCS Total', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'glucose', 'pH', 'BP Diastolic', 'BP Mean', 'BP Systolic'])
    # merged_df.to_csv('processed_train_x.csv', index=False)
    y = merged_df[['hospitaldischargestatus']].values.ravel()
    # np.savetxt('y.csv', y, delimiter=',')
    x = merged_df.drop('hospitaldischargestatus', axis=1)


if __name__ == "__main__":
    main()
