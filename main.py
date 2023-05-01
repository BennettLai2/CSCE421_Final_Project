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

    train_x, test_x, train_y, test_y = split_data(x, y)

    # ###### Your Code Here #######
    # # Add anything you want here

    # ############################

    # ###### Your Code Here #######
    # # Add anything you want here

    # ############################

    
    # skf=StratifiedKFold(n_splits=5, shuffle=False)
    # for train_index, test_index in skf.split(x, y): 
    #     X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    #     Y_train = y[train_index]
    #     Y_test = y[test_index]
        
    #     cv_model = Model(50)  # you can add arguments as needed
    #     acc = cv_model.fit(X_train.drop('patientunitstayid', axis=1), Y_train, X_test.drop('patientunitstayid', axis=1), Y_test)
    #     print(acc)
    model = Model(50)  # you can add arguments as needed
    acc = model.fit(train_x.drop('patientunitstayid', axis=1), train_y, test_x.drop('patientunitstayid', axis=1), test_y)
        

    probas = pd.DataFrame(model.predict_proba(test_x.drop('patientunitstayid', axis=1)), columns=['proba_0', 'proba_1'])
    probas = probas[['proba_1']].values.ravel()
    np.savetxt('probas.csv', probas, delimiter=',')
    # print(test_x)
    patientunitstayid = test_x[['patientunitstayid']].values.ravel()
    unique_ids = np.unique(patientunitstayid)
    mean_proba = [np.max(probas[np.where(patientunitstayid==id)]) for id in unique_ids]

    # # create 2D array with patientunitstayid and mean proba values
    result = np.column_stack((unique_ids, mean_proba))
    np.savetxt('result.csv', result, delimiter=',')
    
    # ---------------------------------------------
    # test_x = load_data("test_x.csv")
    # test_x['hospitaldischargestatus'] = 0
    # test_x = preprocess_x(test_x)
    # test_x['ethnicity_Native American'] = 0
    # test_x.drop('hospitaldischargestatus', axis=1)
    # test_x = test_x.reindex(columns =['patientunitstayid', 'ethnicity_African American', 'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic', 'ethnicity_Native American', 'ethnicity_Other/Unknown', 'gender_Female', 'gender_Male', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'Capillary Refill', 'GCS Total', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'glucose', 'pH', 'BP Diastolic', 'BP Mean', 'BP Systolic'])
    # test_x.to_csv('processed_test_x.csv', index=False)

    # test_x = load_data("processed_test_x.csv")
    # probas = pd.DataFrame(model.predict_proba(test_x.drop(['patientunitstayid'], axis=1)), columns=['proba_0', 'proba_1'])
    # probas = probas[['proba_1']].values.ravel()
    # np.savetxt('probas.csv', probas, delimiter=',')
    # # print(test_x)
    # patientunitstayid = test_x[['patientunitstayid']].values.ravel()
    # unique_ids = np.unique(patientunitstayid)
    # mean_proba = [np.mean(probas[np.where(patientunitstayid==id)]) for id in unique_ids]

    # # # create 2D array with patientunitstayid and mean proba values
    # result = np.column_stack((unique_ids, mean_proba))
    # np.savetxt('result.csv', result, delimiter=',')


if __name__ == "__main__":
    main()
