from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process_and_hold_out(X, Y)

scores_1 = []
scores_2 = []
scores_3 = []

Ks = []

for i in xrange(50):
    K = 2*i + 1
    knn_classifier_1 = KNeighborsClassifier(n_neighbors=K)
    knn_classifier_2 = KNeighborsClassifier(n_neighbors=K)
    knn_classifier_3 = KNeighborsClassifier(n_neighbors=K)
    
    new_score_1 = np.mean(cross_val_score(knn_classifier_1, X_f, Y_no_transform, cv=10))
    new_score_2 = np.mean(cross_val_score(knn_classifier_2, X_f, Y_equal_size, cv=10))
    new_score_3 = np.mean(cross_val_score(knn_classifier_3, X_f, Y_equal_frequency, cv=10))
    
    Ks.append(K)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)
    
plt.xlabel('Valor de K')
plt.ylabel('Acuracia media - 10 fold')
no_transf, = plt.plot(Ks, scores_1, 'b', label='sem transformacao')
equal_length, = plt.plot(Ks, scores_2, 'r', label='tamanhos iguais')
equal_frequency, = plt.plot(Ks, scores_3, 'g', label='frequencias iguais')
plt.legend([no_transf, equal_length, equal_frequency], ['sem transformacao', 'tamanhos iguais', 'frequencias iguais'])
plt.show()
