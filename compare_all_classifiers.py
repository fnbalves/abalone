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

scores_knn = []
scores_mlp = []
scores_svm = []
scores_tree = []


results_knn = []
results_mlp = []
results_svm = []
results_tree = []


for i in xrange(30):
    print i+1, 'of 30'
    [X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process(X, Y)

    knn_classifier = KNeighborsClassifier(n_neighbors=40, weights='distance')
    mlp_classifier = MLPClassifier((10), activation='logistic', solver='lbfgs', learning_rate='constant', max_iter=500)
    svm_classifier = SVC(C = 20, kernel = 'linear')
    tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=25)

    all_test_1 = cross_val_score(knn_classifier, X_f, Y_equal_frequency, cv=3, scoring='f1_macro')
    all_test_2 = cross_val_score(mlp_classifier, X_f, Y_equal_frequency, cv=3, scoring='f1_macro')
    all_test_3 = cross_val_score(svm_classifier, X_f, Y_equal_frequency, cv=3, scoring='f1_macro')
    all_test_4 = cross_val_score(tree_classifier, X_f, Y_equal_frequency, cv=3, scoring='f1_macro')
    
    new_score_1 = np.mean(all_test_1)
    new_score_2 = np.mean(all_test_2)
    new_score_3 = np.mean(all_test_3)
    new_score_4 = np.mean(all_test_4)
    
    scores_knn.append(new_score_1)
    scores_mlp.append(new_score_2)
    scores_svm.append(new_score_3)
    scores_tree.append(new_score_4)

    results_knn.append(all_test_1)
    results_mlp.append(all_test_2)
    results_svm.append(all_test_3)
    results_tree.append(all_test_4)
    
plt.xlabel('Tentativa')
plt.ylabel('F1-score')
knn, = plt.plot(scores_knn, 'b', label='knn')
mlp, = plt.plot(scores_mlp, 'r', label='mlp')
svm, = plt.plot(scores_svm, 'g', label='svm')
tree, = plt.plot(scores_tree, 'k', label='tree')
plt.legend([knn, mlp, svm, tree], ['knn', 'mlp', 'svm', 'tree'])
plt.show()

out_knn = open('result_pickles/scores_knn.pickle', 'wb')
out_mlp = open('result_pickles/scores_mlp.pickle', 'wb')
out_svm = open('result_pickles/scores_svm.pickle', 'wb')
out_tree = open('result_pickles/scores_tree.pickle', 'wb')

pickle.dump(results_knn, out_knn)
pickle.dump(results_mlp, out_mlp)
pickle.dump(results_svm, out_svm)
pickle.dump(results_tree, out_tree)

out_knn.close()
out_mlp.close()
out_svm.close()
out_tree.close()
