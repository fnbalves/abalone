from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train, X_test, Y_test] = pre_process_and_hold_out(X, Y)

accuracies = []

for i in xrange(50):
    K = 2*i + 1
    knn_classifier = KNeighborsRegressor(n_neighbors=K)
    knn_classifier.fit(X_train, Y_train)
    new_accuracy = knn_classifier.score(X_test, Y_test)
    accuracies.append([K, new_accuracy])

plt.xlabel('Valor de K')
plt.ylabel('Acuracia')
x = [a for [a,b] in accuracies]
y = [b for [a,b] in accuracies]
plt.plot(x,y)
plt.show()