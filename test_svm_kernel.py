from util import *
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process(X, Y)
len_train = int(2.0*float(len(X_f))/3.0)
X_f = X_f[:len_train]
Y_no_transform = Y_no_transform[:len_train]
Y_equal_size = Y_equal_size[:len_train]
Y_equal_frequency = Y_equal_frequency[:len_train]

scores_1 = []
scores_2 = []
scores_3 = []
Ks = []

for i in xrange(200):
    c = (5.0*(i + 1))/50.0
    print i+1, 'of 200'
    svm_classifier_1 = SVC(C = c, kernel = 'rbf')
    svm_classifier_2 = SVC(C = c, kernel = 'poly')
    svm_classifier_3 = SVC(C = c, kernel = 'linear')

    new_score_1 = np.mean(cross_val_score(svm_classifier_1, X_f, Y_equal_frequency, cv=3, scoring='f1_macro'))
    new_score_2 = np.mean(cross_val_score(svm_classifier_2, X_f, Y_equal_frequency, cv=3, scoring='f1_macro'))
    new_score_3 = np.mean(cross_val_score(svm_classifier_3, X_f, Y_equal_frequency, cv=3, scoring='f1_macro'))

    Ks.append(c)

    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)

plt.xlabel('Valor de C')
plt.ylabel('F1 score medio - 3 fold')
rbf, = plt.plot(Ks, scores_1, 'b', label='rbf')
poly, = plt.plot(Ks, scores_2, 'r', label='poly 3')
linear, = plt.plot(Ks, scores_3, 'g', label='linear')
plt.legend([rbf, poly, linear], ['rbf', 'poly 3', 'linear'])
plt.show()
