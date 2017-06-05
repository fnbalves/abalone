from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train_1, X_train_2, Y_train, X_test_1, X_test_2, Y_test] = test_custon_transformation(X, Y)
scores_1 = []
scores_2 = []

Ks = []
for i in xrange(50):
    K = 2*i + 1
    knn_classifier_1 = KNeighborsClassifier(n_neighbors=K)
    knn_classifier_2 = KNeighborsClassifier(n_neighbors=K)
    knn_classifier_1.fit(X_train_1, Y_train)
    knn_classifier_2.fit(X_train_2, Y_train)
    
    predictions_1 = knn_classifier_1.predict(X_test_1)
    predictions_2 = knn_classifier_2.predict(X_test_2)
    new_score_1 = f1_score(Y_test, predictions_1, average='macro')
    new_score_2 = f1_score(Y_test, predictions_2, average='macro')
    Ks.append(K)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    
plt.xlabel('Valor de K')
plt.ylabel('F1-score')
no_transf, = plt.plot(Ks, scores_1, 'b', label='sem transformacao')
with_transf, = plt.plot(Ks, scores_2, 'r', label='com transformacao')
plt.legend([no_transf, with_transf], ['sem transformacao', 'com transformacao'])
plt.show()
