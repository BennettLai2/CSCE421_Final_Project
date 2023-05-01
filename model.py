from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Model():
    def __init__(self, n_neighbors: int):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed

        ########################################################################
        self.neigh = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=0)
        self.cols = ['age', 'unitvisitnumber', 'admissionweight', 'GCS Total', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'BP Mean']

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################
        # Fit your model to the training data here

        ########################################################################
        x_train = x_train[self.cols]
        x_val = x_val[self.cols]
        # self.pca = PCA(n_components=3)
        # self.pca.fit(x_train)
        # x_train = self.pca.transform(x_train)
        # x_val = self.pca.transform(x_val)
        self.neigh.fit(x_train, y_train)
        threshold = 0.3
        predicted_proba = self.neigh.predict_proba(x_val)
        predicted = (predicted_proba [:,1] >= threshold).astype('int')

        accuracy = accuracy_score(y_val, predicted)
        return accuracy

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x

        ########################################################################
        x =x[self.cols]
        # x = self.pca.transform(x)
        probas = self.neigh.predict_proba(x)
        return probas