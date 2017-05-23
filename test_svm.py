from util import *
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import decomposition

import matplotlib.pyplot as plt

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

pca = decomposition.PCA(n_components=5)

[X_train, Y_train, X_test, Y_test] = pre_process_and_hold_out(X, Y)
pca.fit(X_train)
n_X_train = pca.transform(X_train)
n_X_test = pca.transform(X_test)

accuracies = []
predictions = []

for i in xrange(200):
    c = (5.0*(i + 1))/50.0
    print i
    svm_classifier = SVR(C = c, kernel = 'linear')
    svm_classifier.fit(n_X_train, Y_train)
    new_accuracy = np.sqrt(mean_error(svm_classifier, n_X_test, Y_test))
    print 'Erro medio', new_accuracy
    accuracies.append([c, new_accuracy])
    predictions = svm_classifier.predict(n_X_test)

plt.xlabel('Valor de C')
plt.ylabel('Raiz do erro medio')
plt.scatter(predictions, Y_test)
plt.show()
#x = [a for [a,b] in accuracies]
#y = [b for [a,b] in accuracies]
#plt.plot(x,y)
#plt.show()