import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(x_path):
    # Your code here
    return pd.read_csv(x_path, 
                       dtype={'admissionheight': float, 
                              'admissionweight': float, 
                              'age': str, 
                              'cellattributevalue': str, 
                              'celllabel': str, 
                              'ethnicity': str, 
                              'gender': str, 
                              'labmeasurenamesystem': str, 
                              'labname': str, 
                              'labresult': float, 
                              'nursingchartcelltypevalname': str, 
                              'nursingchartvalue': str, 
                              'offset': float, 
                              'patientunitstayid': int, 
                              'unitvisitnumber': float})


def split_data(x, y, split=0.8):
    # Your code here
    train_x, train_y, test_x, test_y = train_test_split(x, y, test_size = 1-split)
    return train_x, train_y, test_x, test_y


def preprocess_x(df):
    # Your code here
    # sort data by the following
    df = df.sort_values(by=['patientunitstayid', 'age', 'offset', 'nursingchartcelltypevalname'])
    # forward filling the height and weight by mean of age, ethnicity, and gender
    df['admissionheight'] = df.groupby(['age', 'ethnicity', 'gender'])['admissionheight'].transform(lambda x: x.fillna(x.mean()))
    df['admissionweight'] = df.groupby(['age', 'ethnicity', 'gender'])['admissionweight'].transform(lambda x: x.fillna(x.mean()))
    # drop missing lab results
    mask = (~df['labname'].isnull()) & (df['labresult'].isnull())
    df.drop(index=df.loc[mask].index, inplace=True)
    # one-hot encode for celllabel
    df['cellattributevalue'] = df['cellattributevalue'].replace({'normal': 0, 
                                                                '< 2 seconds': 1, 
                                                                '> 2 seconds': 2, 
                                                                'feet': 3, 
                                                                'hands': 4})
    # replace missing GCS for filling
    df = df.replace('Unable to score due to medication', pd.NaT)
    cols = ['admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'unitvisitnumber', 'nursingchartvalue']
    df.loc[:,cols] = df.loc[:,cols].ffill()
    # changing age to int dtype
    df['age'] = df['age'].replace(['> 89'], '89')
    df['age'] = df['age'].astype(int)

    # df.to_csv('pre.csv')
    
    condensed_df1 = df.pivot_table(
        index=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus'],
        columns=['nursingchartcelltypevalname'],
        values=['nursingchartvalue']
    ).reset_index()
    
    condensed_df2 = df.pivot_table(
        index=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus'],
        columns=['labname'],
        values=['labresult']
    ).reset_index()

    condensed_df3 = df.pivot_table(
        index=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus'],
        columns=['celllabel'],
        values=['cellattributevalue']
    ).reset_index()
    
    condensed_df4 = pd.merge(condensed_df1, condensed_df2, on=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus'], how='outer')
    condensed_df = pd.merge(condensed_df3, condensed_df4, on=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus'], how='outer')
    
    condensed_df = condensed_df.sort_values(by=['patientunitstayid', 'offset'])
    
    condensed_df.columns = ['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus', 'Capillary Refill', 'GCS Total', 'Heart Rate', 'Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Mean', 'Non-Invasive BP Systolic', 'O2 Saturation', 'Respiratory Rate', 'glucose', 'pH']
    condensed_df.to_csv('processed_train_x.csv', index=False)
    print(condensed_df.columns)
    return condensed_df
