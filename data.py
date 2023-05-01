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
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 1-split, shuffle=False)
    return train_x, test_x, train_y, test_y


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
    df['celllabel'] = df['celllabel'].fillna('Capillary Refill')
    df['cellattributevalue'] = df['cellattributevalue'].fillna(-1)

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
        values=['cellattributevalue'], 
        aggfunc='min'
    ).reset_index()
    
    condensed_df4 = pd.merge(condensed_df1, condensed_df2, on=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus'], how='outer')
    condensed_df = pd.merge(condensed_df3, condensed_df4, on=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus'], how='outer')
    
    condensed_df = condensed_df.sort_values(by=['patientunitstayid', 'offset'])
    
    condensed_df.columns = ['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'hospitaldischargestatus', 'Capillary Refill', 'GCS Total', 'Heart Rate', 'Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Mean', 'Non-Invasive BP Systolic', 'O2 Saturation', 'Respiratory Rate', 'glucose', 'pH']
    
    # condensed_df['Capillary Refill'] = condensed_df['Capillary Refill'].ffill()
    # condensed_df['Capillary Refill'] = condensed_df['Capillary Refill'].fillna(0, inplace=True)
    
    condensed_df['Capillary Refill'] = condensed_df['Capillary Refill'].replace(-1, pd.NaT)
    condensed_df['Capillary Refill'] = condensed_df.groupby('patientunitstayid')['Capillary Refill'].ffill().fillna(0)
    condensed_df['Heart Rate'] = condensed_df.groupby('patientunitstayid')['Heart Rate'].ffill().bfill()
    condensed_df['Heart Rate'] = condensed_df['Heart Rate'].fillna(condensed_df['Heart Rate'])
    condensed_df['O2 Saturation'] = condensed_df.groupby('patientunitstayid')['O2 Saturation'].ffill().bfill()
    condensed_df['O2 Saturation'] = condensed_df['O2 Saturation'].fillna(100)
    condensed_df['Respiratory Rate'] = condensed_df.groupby('patientunitstayid')['Respiratory Rate'].ffill().bfill()
    condensed_df['Respiratory Rate'] = condensed_df['Respiratory Rate'].fillna(18)
    condensed_df['glucose'] = condensed_df.groupby('patientunitstayid')['glucose'].ffill().bfill()
    condensed_df['glucose'] = condensed_df['glucose'].fillna(86.6)
    condensed_df['pH'] = condensed_df.groupby('patientunitstayid')['pH'].ffill().bfill()
    condensed_df['pH'] = condensed_df['pH'].fillna(7.4)
    
    condensed_df['GCS Total'] = condensed_df.groupby('patientunitstayid')['GCS Total'].ffill().fillna(15)

    combined_df = condensed_df[['Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Mean', 'Non-Invasive BP Systolic']].copy()
    combined_df['BP Diastolic'] = combined_df['Invasive BP Diastolic'].combine_first(combined_df['Non-Invasive BP Diastolic'])
    combined_df['BP Mean'] = combined_df['Invasive BP Mean'].combine_first(combined_df['Non-Invasive BP Mean'])
    combined_df['BP Systolic'] = combined_df['Invasive BP Systolic'].combine_first(combined_df['Non-Invasive BP Systolic'])
    combined_df = combined_df[['BP Diastolic', 'BP Mean', 'BP Systolic']]

    condensed_df.drop(['Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic',
                       'Non-Invasive BP Mean', 'Non-Invasive BP Systolic'], axis=1, inplace=True)
    condensed_df = pd.concat([condensed_df, combined_df], axis=1)
    condensed_df['BP Diastolic'] = condensed_df.groupby('patientunitstayid')['BP Diastolic'].ffill().bfill()
    condensed_df['BP Diastolic'] = condensed_df['BP Diastolic'].fillna(condensed_df['BP Diastolic'])    
    condensed_df['BP Systolic'] = condensed_df.groupby('patientunitstayid')['BP Systolic'].ffill().bfill()
    condensed_df['BP Systolic'] = condensed_df['BP Systolic'].fillna(condensed_df['BP Systolic'])   
    condensed_df['BP Mean'] = condensed_df['BP Mean'].fillna(condensed_df['BP Diastolic'].shift(1) + (condensed_df['BP Systolic'].shift(-1) - condensed_df['BP Diastolic'].shift(1))/3)
    condensed_df=condensed_df.ffill().bfill()
    categorical = condensed_df.select_dtypes(exclude=['int64', 'float64'])
    numerical = condensed_df.select_dtypes(include=['int64', 'float64'])
    cate_dummies = pd.get_dummies(categorical)
    condensed_df = pd.concat([cate_dummies, numerical], axis=1)

    return condensed_df
