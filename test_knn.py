from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train, X_test, Y_test] = pre_process_and_hold_out(X, Y)

#eliminar var 5, 7, 8, 9
(num_elems, dimensions) = np.shape(X_train)
num_elems_test = np.shape(X_test)[0]

selected_colums = range(dimensions)
#selected_colums.remove(0)
#selected_colums.remove(1)
#selected_colums.remove(2)
#selected_colums.remove(5)
#selected_colums.remove(7)
#selected_colums.remove(8)
#selected_colums.remove(9)

n_X_train = X_train[np.ix_(range(num_elems), selected_colums)]
n_X_test = X_test[np.ix_(range(num_elems_test), selected_colums)]

accuracies = []
predictions = []
for i in xrange(50):
    K = 2*i + 1
    knn_classifier = KNeighborsRegressor(n_neighbors=K, weights='distance')
    knn_classifier.fit(n_X_train, Y_train)
    new_accuracy = np.sqrt(mean_error(knn_classifier, n_X_test, Y_test))#knn_classifier.score(n_X_test, Y_test)
    print 'Mean error', new_accuracy

    accuracies.append([K, new_accuracy])
    predictions = knn_classifier.predict(n_X_test)

plt.xlabel('Valor de K')
plt.ylabel('Raiz do erro medio')
plt.scatter(predictions, Y_test)
plt.show()
#x = [a for [a,b] in accuracies]
#y = [b for [a,b] in accuracies]
#plt.plot(x,y)
#plt.show()