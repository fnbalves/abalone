from util import *
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train_1, Y_train_2, Y_train_3, X_test, Y_test_1, Y_test_2, Y_test_3] = pre_process_and_hold_out(X, Y)

scores_1 = []
scores_2 = []
scores_3 = []
Ks = []

for i in xrange(200):
    c = (5.0*(i + 1))/50.0
    print i
    svm_classifier_1 = SVC(C = c, kernel = 'rbf')
    svm_classifier_2 = SVC(C = c, kernel = 'rbf')
    svm_classifier_3 = SVC(C = c, kernel = 'rbf')
    svm_classifier_1.fit(X_train, Y_train_1)
    svm_classifier_2.fit(X_train, Y_train_2)
    svm_classifier_3.fit(X_train, Y_train_3)
    
    predictions_1 = svm_classifier_1.predict(X_test)
    predictions_2 = svm_classifier_2.predict(X_test)
    predictions_3 = svm_classifier_3.predict(X_test)
    new_score_1 = f1_score(Y_test_1, predictions_1, average='macro')
    new_score_2 = f1_score(Y_test_2, predictions_2, average='macro')
    new_score_3 = f1_score(Y_test_3, predictions_3, average='macro')
    Ks.append(c)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)

plt.xlabel('Valor de C')
plt.ylabel('F1-score')
no_transf, = plt.plot(Ks, scores_1, 'b', label='sem transformacao')
equal_length, = plt.plot(Ks, scores_2, 'r', label='tamanhos iguais')
equal_frequency, = plt.plot(Ks, scores_3, 'g', label='frequencias iguais')
plt.legend([no_transf, equal_length, equal_frequency], ['sem transformacao', 'tamanhos iguais', 'frequencias iguais'])
plt.show()
