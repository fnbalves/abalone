from util import *
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train, X_test, Y_test] = pre_process_and_hold_out(X, Y)

accuracies = []

for i in xrange(50):
    c = (5.0*(i + 1))/500.0
    print i
    svm_classifier = MLPRegressor((10), activation='relu', solver='lbfgs', alpha=c)
    svm_classifier.fit(X_train, Y_train)
    new_accuracy = svm_classifier.score(X_test, Y_test)
    accuracies.append([c, new_accuracy])

plt.xlabel('Valor de C')
plt.ylabel('Acuracia')
x = [a for [a,b] in accuracies]
y = [b for [a,b] in accuracies]
plt.plot(x,y)
plt.show()