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

def main():

    x = load_data("train_x.csv")
    x = preprocess_x(x)
    x.to_csv('processed_train_x.csv')
    y = load_data('train_y.csv')
    
    y = pd.DataFrame({'hospitaldischargestatus': x['patientunitstayid'].map(y.set_index('patientunitstayid')['hospitaldischargestatus'])})
    
    train_x, test_x, train_y, test_y = split_data(x, y)

    # -------------------------
    print("SKF:")
    mean_score = 0
    skf=StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(x, y): 
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
        
        cv_model = Model()
        acc = cv_model.fit(X_train, Y_train, X_test, Y_test)
        print(acc)
        mean_score += acc
    print("Average ROC: ", mean_score / 5)
    model = Model()
    acc = model.fit(train_x, train_y, test_x, test_y)
    # print()
    # print("Full Dataset:")
    # print(acc)

    test_x = load_data('test_x.csv')
    test_x = preprocess_x(test_x)
    test_x.to_csv('processed_test_x.csv')
    pred = model.predict_proba(test_x)
    pred.to_csv('pred.csv', index=False)

if __name__ == "__main__":
    main()
