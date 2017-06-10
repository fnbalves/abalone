from util import *
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
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

scores_knn = []
scores_mlp = []
scores_svm = []
scores_tree = []
scores_combined = []

for i in xrange(20):
    print i
    [X_train, Y_train_1, Y_train_2, Y_train_3, X_test, Y_test_1, Y_test_2, Y_test_3] = pre_process_and_hold_out(X, Y)

    knn_classifier = KNeighborsClassifier(n_neighbors=40, weights='distance')
    mlp_classifier = MLPClassifier((10), activation='tanh', solver='lbfgs', learning_rate='invscaling', max_iter=500)
    svm_classifier = SVC(C = 7, kernel = 'linear')
    tree_classifier = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
    combined = CombinedClassifier()

    knn_classifier.fit(X_train, Y_train_3)
    mlp_classifier.fit(X_train, Y_train_3)
    svm_classifier.fit(X_train, Y_train_3)
    tree_classifier.fit(X_train, Y_train_3)
    combined.fit(X_train, Y_train_3)
    
    predictions_1 = knn_classifier.predict(X_test)
    predictions_2 = mlp_classifier.predict(X_test)
    predictions_3 = svm_classifier.predict(X_test)
    predictions_4 = tree_classifier.predict(X_test)
    predictions_5 = combined.predict(X_test)
    
    new_score_1 = f1_score(Y_test_3, predictions_1, average='macro')
    new_score_2 = f1_score(Y_test_3, predictions_2, average='macro')
    new_score_3 = f1_score(Y_test_3, predictions_3, average='macro')
    new_score_4 = f1_score(Y_test_3, predictions_4, average='macro')
    new_score_5 = f1_score(Y_test_3, predictions_5, average='macro')
    
    scores_knn.append(new_score_1)
    scores_mlp.append(new_score_2)
    scores_svm.append(new_score_3)
    scores_tree.append(new_score_4)
    scores_combined.append(new_score_5)
    
plt.xlabel('Tentativa')
plt.ylabel('F1-score')
knn, = plt.plot(scores_knn, 'b', label='knn')
mlp, = plt.plot(scores_mlp, 'r', label='mlp')
svm, = plt.plot(scores_svm, 'g', label='svm')
tree, = plt.plot(scores_tree, 'k', label='tree')
combined, = plt.plot(scores_combined, 'y', label='combined')
plt.legend([knn, mlp, svm, tree, combined], ['knn', 'mlp', 'svm', 'tree', 'combined'])
plt.show()

out_knn = open('result_pickles/scores_knn.pickle', 'wb')
out_mlp = open('result_pickles/scores_mlp.pickle', 'wb')
out_svm = open('result_pickles/scores_svm.pickle', 'wb')
out_tree = open('result_pickles/scores_tree.pickle', 'wb')

pickle.dump(scores_knn, out_knn)
pickle.dump(scores_mlp, out_mlp)
pickle.dump(scores_svm, out_svm)
pickle.dump(scores_tree, out_tree)

out_knn.close()
out_mlp.close()
out_svm.close()
out_tree.close()
