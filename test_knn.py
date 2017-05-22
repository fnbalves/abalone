from util import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train, X_test, Y_test] = hold_out(X, Y)

accuracies = []

for K in xrange(10):
    knn_classifier = KNeighborsClassifier(n_neighbors=K)
    knn_classifier.fit(X_train, Y_train)
    new_accuracy = knn_classifier.score(X_test, Y_test)
    accuracies.append([K, new_accuracy])

plt.xlabel('Valor de K')
plt.ylabel('Acuracia')
x = [x for [x,y] in accuracies]
y = [y for [x,y] in accuracies]
plt.plot(x,y)
plt.show()