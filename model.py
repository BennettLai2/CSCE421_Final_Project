from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed

        ########################################################################
        self.forest = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight="balanced", random_state=0)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################
        # Fit your model to the training data here

        ########################################################################
        x_train, y_train, x_val, y_val = x_train.drop('patientunitstayid', axis=1), y_train.values.ravel(), x_val.drop('patientunitstayid', axis=1), y_val.values.ravel()
        
        self.forest.fit(x_train, y_train)

        y_pred_proba = self.forest.predict_proba(x_val)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_val,  y_pred_proba)

        #create ROC curve
        plt.plot(fpr,tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        return roc_auc_score(y_val, self.forest.predict_proba(x_val)[:,1])
        

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x

        ########################################################################
        patientunitstayid = x['patientunitstayid'].values.ravel()
        x = x.drop('patientunitstayid', axis = 1)
       
        probas = self.forest.predict_proba(x)
        probas = probas[:, 1]
        unique_ids = np.unique(patientunitstayid)
        mean_proba = np.array([np.mean(probas[np.where(patientunitstayid==id)]) for id in unique_ids])

        df = pd.DataFrame({'patientunitstayid': unique_ids, 'hospitaldischargestatus': mean_proba})
        df['patientunitstayid'] = df['patientunitstayid'].astype(int)
        df['hospitaldischargestatus'] = df['hospitaldischargestatus'].astype(float)
        return df