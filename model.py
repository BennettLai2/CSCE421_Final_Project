from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.svm import SVC

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
        self.pca = PCA(n_components=7)
        self.pca.fit(x_train)
        x_train = self.pca.transform(x_train)
        x_val = self.pca.transform(x_val)
        self.neigh.fit(x_train, y_train)
        return self.neigh.score(x_val, y_val)

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x

        ########################################################################
        x = self.pca.transform(x)
        probas = self.neigh.predict_proba(x)
        return probas