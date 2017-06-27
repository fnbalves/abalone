from util import *
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from CombinedClassifier import *
import matplotlib.pyplot as plt
import pickle

import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process(X, Y)

all_data = [[X_f[i], Y_equal_frequency[i]] for i, x in enumerate(X_f)]
random.shuffle(all_data)
len_train = int(2.0*float(len(all_data))/3.0)
train = all_data[:len_train]
test = all_data[len_train:]
X_train = [x for [x,y] in train]
Y_train = [y for [x,y] in train]
X_test = [x for [x,y] in test]
Y_test = [y for [x,y] in test]

knn_classifier = KNeighborsClassifier(n_neighbors=40, weights='distance')
mlp_classifier = MLPClassifier((10), activation='logistic', solver='lbfgs', learning_rate='constant', max_iter=500)
svm_classifier = SVC(C = 20, kernel = 'linear')
tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=25)

knn_classifier.fit(X_train, Y_train)    
mlp_classifier.fit(X_train, Y_train)
svm_classifier.fit(X_train, Y_train)
tree_classifier.fit(X_train, Y_train)

predictions_1 = knn_classifier.predict(X_test)
predictions_2 = mlp_classifier.predict(X_test)
predictions_3 = svm_classifier.predict(X_test)
predictions_4 = tree_classifier.predict(X_test)

classifiers = [['knn', predictions_1], ['mlp', predictions_2], ['svm', predictions_3], ['tree', predictions_4]]
len_classifiers = len(classifiers)
for i in xrange(len_classifiers):
    for j in xrange(i+1, len_classifiers):
        c = np.cov(classifiers[i][1], classifiers[j][1])
        print 'covariance', classifiers[i][0], 'x', classifiers[j][0], ':', c[0,1]/np.sqrt(c[0,0]*c[1,1])
