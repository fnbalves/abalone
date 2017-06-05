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

[X_train, Y_train_1, Y_train_2, Y_train_3, X_test, Y_test_1, Y_test_2, Y_test_3] = pre_process_and_hold_out(X, Y)

scores_1 = []
scores_2 = []

Ks = []
for i in xrange(50):
    K = 2*i + 1
    knn_classifier_1 = KNeighborsClassifier(n_neighbors=K, weights='uniform')
    knn_classifier_2 = KNeighborsClassifier(n_neighbors=K, weights='distance')
    knn_classifier_1.fit(X_train, Y_train_3)
    knn_classifier_2.fit(X_train, Y_train_3)
    
    predictions_1 = knn_classifier_1.predict(X_test)
    predictions_2 = knn_classifier_2.predict(X_test)
    
    new_score_1 = f1_score(Y_test_3, predictions_1, average='macro')
    new_score_2 = f1_score(Y_test_3, predictions_2, average='macro')
    
    Ks.append(K)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    
plt.xlabel('Valor de K')
plt.ylabel('F1-score')
uniform, = plt.plot(Ks, scores_1, 'b', label='uniform')
distance, = plt.plot(Ks, scores_2, 'r', label='distance')

plt.legend([uniform, distance], ['uniform', 'distance'])
plt.show()
