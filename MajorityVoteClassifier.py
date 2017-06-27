import pandas as pd
import numpy as np
import math as m
import pickle
import warnings
import random
import operator

#ignore warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class MajorityVoteClassifier:
    def __init__(self):
        
        self.knn_classifier = KNeighborsClassifier(n_neighbors=40, weights='distance')
        self.mlp_classifier = MLPClassifier((10), activation='logistic', solver='lbfgs', learning_rate='constant', max_iter=500)
        self.svm_classifier = SVC(C = 20, kernel = 'linear')
        self.tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=18)

    def fit(self, X, Y):
        self.knn_classifier.fit(X, Y)
        self.mlp_classifier.fit(X, Y)
        self.svm_classifier.fit(X, Y)
        self.tree_classifier.fit(X, Y)
        
    def predict(self, x):
        predict_knn = self.knn_classifier.predict(x)
        predict_mlp = self.mlp_classifier.predict(x)
        predict_svm = self.svm_classifier.predict(x)
        predict_tree = self.tree_classifier.predict(x)
        
        all_predictions = predict_knn + predict_mlp + predict_svm + predict_tree
        classes = sorted(set(all_predictions))

        prediction = []
        for i, p in enumerate(predict_knn):
            result_dic = {}
            for (index, c) in enumerate(classes):
                result_dic[index] = 0

            result_dic[predict_knn[i]] += 1
            result_dic[predict_mlp[i]] += 1
            result_dic[predict_svm[i]] += 1
            result_dic[predict_tree[i]] += 1
            votes = sorted(result_dic.items(), key=operator.itemgetter(1), reverse=True)
            prediction.append(votes[0][0])
        return prediction
