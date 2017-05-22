from util import *
from sklearn.svm import SVC
from sklearn.svm import SVR

import matplotlib.pyplot as plt

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train, X_test, Y_test] = pre_process_and_hold_out(X, Y)

accuracies = []

for i in xrange(50):
    c = (5.0*(i + 1))/50.0
    print i
    svm_classifier = SVR(C = c, kernel = 'rbf')
    svm_classifier.fit(X_train, Y_train)
    new_accuracy = svm_classifier.score(X_test, Y_test)
    accuracies.append([c, new_accuracy])

plt.xlabel('Valor de C')
plt.ylabel('Acuracia')
x = [a for [a,b] in accuracies]
y = [b for [a,b] in accuracies]
plt.plot(x,y)
plt.show()