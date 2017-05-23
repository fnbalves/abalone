from util import *
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
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

errors = []
predictions = []
best_error = 10000
for i in xrange(20):
    c = i + 1
    print i
    mlp_classifier = MLPClassifier((c), activation='relu', solver='lbfgs', learning_rate='constant')
    mlp_classifier.fit(n_X_train, Y_train)
    new_error = np.sqrt(mean_error(mlp_classifier, n_X_test, Y_test))
    print 'Mean error', new_error
    errors.append([c, new_error])
    if new_error < best_error:
        best_error = new_error
        predictions = mlp_classifier.predict(n_X_test)

#plt.xlabel('Valor de C')
#plt.ylabel('Raiz do erro medio')
#x = [a for [a,b] in errors]
#y = [b for [a,b] in errors]
#plt.plot(x,y)
#plt.show()
plt.xlabel('Valores do MLP')
plt.ylabel('Valores originais')
plt.scatter(predictions, Y_test)
plt.show()
