from util import *
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from CombinedClassifier import *
import matplotlib.pyplot as plt
import pickle
import time
import random

import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

times_train_knn = []
times_evaluate_knn = []
times_train_mlp = []
times_evaluate_mlp = []
times_train_svm = []
times_evaluate_svm = []
times_train_tree = []
times_evaluate_tree = []

plot_train = False

Ts = []
for i in xrange(20):
    print i+1, 'of 20'
    Ts.append(i)
    
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

    init = time.clock()
    knn_classifier.fit(X_train, Y_train)
    times_train_knn.append(time.clock() - init)
    
    init = time.clock()
    mlp_classifier.fit(X_train, Y_train)
    times_train_mlp.append(time.clock() - init)

    init = time.clock()
    svm_classifier.fit(X_train, Y_train)
    times_train_svm.append(time.clock() - init)

    init = time.clock()
    tree_classifier.fit(X_train, Y_train)
    times_train_tree.append(time.clock() - init)

    init = time.clock()
    predictions_1 = knn_classifier.predict(X_test)
    times_evaluate_knn.append(time.clock() - init)

    init = time.clock()
    predictions_2 = mlp_classifier.predict(X_test)
    times_evaluate_mlp.append(time.clock() - init)

    init = time.clock()
    predictions_3 = svm_classifier.predict(X_test)
    times_evaluate_svm.append(time.clock() - init)

    init = time.clock()
    predictions_4 = tree_classifier.predict(X_test)
    times_evaluate_tree.append(time.clock() - init)

if plot_train:
    plt.xlabel('Tentativa')
    plt.ylabel('Tempo de treinamento')
    knn, = plt.plot(Ts, times_train_knn, 'b', label='knn')
    mlp, = plt.plot(Ts, times_train_mlp, 'r', label='mlp')
    svm, = plt.plot(Ts, times_train_svm, 'g', label='svm')
    tree, = plt.plot(Ts, times_train_tree, 'k', label='tree')
    plt.legend([knn, mlp, svm, tree], ['knn', 'mlp', 'svm', 'tree'])
    plt.show()
else:
    plt.xlabel('Tentativa')
    plt.ylabel('Tempo de uso')
    knn, = plt.plot(Ts, times_evaluate_knn, 'b', label='knn')
    mlp, = plt.plot(Ts, times_evaluate_mlp, 'r', label='mlp')
    svm, = plt.plot(Ts, times_evaluate_svm, 'g', label='svm')
    tree, = plt.plot(Ts, times_evaluate_tree, 'k', label='tree')
    plt.legend([knn, mlp, svm, tree], ['knn', 'mlp', 'svm', 'tree'])
    plt.show()

