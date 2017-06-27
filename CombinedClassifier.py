from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class CombinedClassifier(object):
    def __init__(self):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=40, weights='distance')
        self.mlp_classifier = MLPClassifier((10), activation='logistic', solver='lbfgs', learning_rate='constant', max_iter=500)
        self.svm_classifier = SVC(C = 20, kernel = 'linear')
        self.tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=18)
        self.last_classifier = MLPClassifier((10), activation='logistic', solver='lbfgs', learning_rate='constant', max_iter=500)
        
    def fit(self, X_train, Y_train):
        self.knn_classifier.fit(X_train, Y_train)
        self.mlp_classifier.fit(X_train, Y_train)
        self.svm_classifier.fit(X_train, Y_train)
        self.tree_classifier.fit(X_train, Y_train)
        Y_knn = self.knn_classifier.predict(X_train)
        Y_mlp = self.mlp_classifier.predict(X_train)
        Y_svm = self.svm_classifier.predict(X_train)
        Y_tree = self.tree_classifier.predict(X_train)
        Y_final = np.array([0.5*np.array((y_knn, Y_mlp[i], Y_svm[i], Y_tree[i])) for i,y_knn in enumerate(Y_knn)])
        
        Y_final = Y_final
        self.last_classifier.fit(Y_final, Y_train)
        
    def predict(self, X):
        Y_knn = self.knn_classifier.predict(X)
        Y_mlp = self.mlp_classifier.predict(X)
        Y_svm = self.svm_classifier.predict(X)
        Y_tree = self.tree_classifier.predict(X)

        Y_final = np.array([np.array((y_knn, Y_mlp[i], Y_svm[i], Y_tree[i])) for i,y_knn in enumerate(Y_knn)])
        
        Y_final = Y_final*0.5

        return self.last_classifier.predict(Y_final)
