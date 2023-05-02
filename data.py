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
    df['age'] = df['age'].replace(['> 89'], '89')

    df['cellattributevalue'] = df['cellattributevalue'].replace({'normal': 0, 
                                                                '< 2 seconds': 1, 
                                                                '> 2 seconds': 2, 
                                                                'feet': 3, 
                                                                'hands': 4})
    df['ethnicity'] = df['ethnicity'].replace({'African American': 0, 
                                               	'Asian': 1, 	
                                                'Caucasian': 2, 	
                                                'Hispanic': 3, 	
                                                'Native American': 4, 	
                                                'Other/Unknown': 5})
    df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1})

    mask = (~df['labname'].isnull()) & (df['labresult'].isnull())
    df.drop(index=df.loc[mask].index, inplace=True)

    df = df.replace('Unable to score due to medication', pd.NaT)
    mask = (~df['nursingchartcelltypevalname'].isnull()) & (df['nursingchartvalue'].isnull())
    df.drop(index=df.loc[mask].index, inplace=True)

    cols = ['admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'unitvisitnumber']
    for col in cols: 
        df[col] = df.groupby('patientunitstayid')[col].ffill()

    df['ethnicity'] = df['ethnicity'].fillna(5)
    df['gender'] = df['gender'].fillna(3)

    for col in cols: 
        mean = df[col].apply(pd.to_numeric, errors='coerce').mean()
        df[col] = df.groupby('patientunitstayid')[col].fillna(mean)

    df['celllabel'] = df['celllabel'].fillna('Capillary Refill')
    df['cellattributevalue'] = df['cellattributevalue'].fillna(-1)

    condensed_df1 = df.pivot_table(
        index=['patientunitstayid', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'unitvisitnumber'],
        columns=['nursingchartcelltypevalname'],
        values=['nursingchartvalue']
    ).reset_index()
    
    condensed_df2 = df.pivot_table(
        index=['patientunitstayid', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'unitvisitnumber'],
        columns=['labname'],
        values=['labresult']
    ).reset_index()

    condensed_df3 = df.pivot_table(
        index=['patientunitstayid', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender', 'unitvisitnumber'],
        columns=['celllabel'],
        values=['cellattributevalue'], 
        aggfunc='min'
    ).reset_index()
    
    condensed_df4 = pd.merge(condensed_df1, condensed_df2, on=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender'], how='outer')
    condensed_df = pd.merge(condensed_df3, condensed_df4, on=['patientunitstayid', 'unitvisitnumber', 'offset', 'admissionheight', 'admissionweight', 'age', 'ethnicity', 'gender'], how='outer')
    
    df = condensed_df.sort_values(by=['patientunitstayid', 'offset'])
    df.columns = ["patientunitstayid","offset","admissionheight","admissionweight","age","ethnicity","gender","unitvisitnumber","Capillary Refill","GCS Total","Heart Rate","Invasive BP Diastolic","Invasive BP Mean","Invasive BP Systolic","Non-Invasive BP Diastolic","Non-Invasive BP Mean","Non-Invasive BP Systolic","O2 Saturation","Respiratory Rate","glucose","pH"]
    
    df['Capillary Refill'] = df['Capillary Refill'].replace(-1, pd.NaT)
    combined_df = df[['Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Mean', 'Non-Invasive BP Systolic']].copy()
    combined_df['BP Diastolic'] = combined_df['Invasive BP Diastolic'].combine_first(combined_df['Non-Invasive BP Diastolic'])
    combined_df['BP Mean'] = combined_df['Invasive BP Mean'].combine_first(combined_df['Non-Invasive BP Mean'])
    combined_df['BP Systolic'] = combined_df['Invasive BP Systolic'].combine_first(combined_df['Non-Invasive BP Systolic'])
    combined_df = combined_df[['BP Diastolic', 'BP Mean', 'BP Systolic']]

    df.drop(['Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic',
                       'Non-Invasive BP Mean', 'Non-Invasive BP Systolic'], axis=1, inplace=True)
    df = pd.concat([df, combined_df], axis=1)
    df['BP Diastolic'] = df.groupby('patientunitstayid')['BP Diastolic'].ffill().bfill()
    df['BP Diastolic'] = df['BP Diastolic'].fillna(df['BP Diastolic'].mean())    
    df['BP Systolic'] = df.groupby('patientunitstayid')['BP Systolic'].ffill().bfill()
    df['BP Systolic'] = df['BP Systolic'].fillna(df['BP Systolic'].mean())   
    df['BP Mean'] = df['BP Mean'].fillna(df['BP Diastolic'].shift(1) + (df['BP Systolic'].shift(-1) - df['BP Diastolic'].shift(1))/3)
    
    cols = ["Capillary Refill","GCS Total","Heart Rate","O2 Saturation","Respiratory Rate","glucose","pH"]
    vals = [0, 15, 80, 100, 18, 86.5, 7.4]
    for col in cols: 
        df[col] = df.groupby('patientunitstayid')[col].ffill().bfill()
    for i in range(len(cols)): 
        df[cols[i]] = df.groupby('patientunitstayid')[cols[i]].fillna(vals[i])

    df = df.ffill().bfill()

    
    df = df.apply(pd.to_numeric, errors='coerce')
    df_mean = df.mean()
    df_max = df.groupby('patientunitstayid', as_index=False).max()
    df_min = df.groupby('patientunitstayid', as_index=False).min()
    cols = ['Capillary Refill','GCS Total','Heart Rate','O2 Saturation','Respiratory Rate','glucose','pH','BP Diastolic','BP Mean','BP Systolic']
    df_max[cols] = df_max[cols].subtract(df_mean[cols].values).abs()
    df_min[cols] = df_min[cols].subtract(df_mean[cols].values).abs()
    df = pd.DataFrame(np.maximum(df_max.values, df_min.values), columns=df_max.columns)

    return df
