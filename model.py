from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import pandas as pd

class Model():
    def __init__(self, n_neighbors: int):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed

        ########################################################################
        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################
        # Fit your model to the training data here

        ########################################################################
        x_train, y_train, x_val, y_val = x_train.drop('patientunitstayid', axis=1), y_train.values.ravel(), x_val.drop('patientunitstayid', axis=1), y_val.values.ravel()
        self.cols = x_train.columns
        self.cols = ['age', 'unitvisitnumber', 'offset', 'GCS Total', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'BP Mean']
        x_train = x_train[self.cols]
        x_val = x_val[self.cols]
        # self.pca = PCA(n_components=5)
        # self.pca.fit(x_train)
        # x_train = self.pca.transform(x_train)
        # x_val = self.pca.transform(x_val)
        self.neigh.fit(x_train, y_train)
        return self.neigh.score(x_val, y_val)

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x

        ########################################################################
        patientunitstayid = x['patientunitstayid'].values.ravel()
        x.drop('patientunitstayid', axis = 1)
        x =x[self.cols]
        # x = self.pca.transform(x)
        probas = self.neigh.predict_proba(x)
        probas = probas[:, 1]
        unique_ids = np.unique(patientunitstayid)
        mean_proba = np.array([np.mean(probas[np.where(patientunitstayid==id)]) for id in unique_ids])

        df = pd.DataFrame({'patientunitstayid': unique_ids, 'hospitaldischargestatus': mean_proba})
        return df