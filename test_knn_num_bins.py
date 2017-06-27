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


scores_1 = []
scores_2 = []
scores_3 = []

Ks = []
fixed_05 = []

for i in xrange(1, 29):
    K = i + 1
    
    print i+1, 'of 29'
    try:
        [X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pd_pre_process(X, Y, K)
        len_train = int(2.0*float(len(X_f))/3.0)
        X_f = X_f[:len_train]
        Y_no_transform = Y_no_transform[:len_train]
        Y_equal_size = Y_equal_size[:len_train]
        Y_equal_frequency = Y_equal_frequency[:len_train]

        knn_classifier_1 = KNeighborsClassifier(n_neighbors=7)
        knn_classifier_2 = KNeighborsClassifier(n_neighbors=7)
        knn_classifier_3 = KNeighborsClassifier(n_neighbors=7)
        
        new_score_1 = np.mean(cross_val_score(knn_classifier_1, X_f, Y_no_transform, cv=3, scoring='f1_macro'))
        new_score_2 = np.mean(cross_val_score(knn_classifier_2, X_f, Y_equal_size, cv=3, scoring='f1_macro'))
        new_score_3 = np.mean(cross_val_score(knn_classifier_3, X_f, Y_equal_frequency, cv=3, scoring='f1_macro'))
        
        Ks.append(K)
        fixed_05.append(0.5)
        
        scores_1.append(new_score_1)
        scores_2.append(new_score_2)
        scores_3.append(new_score_3)
    except:
        break
    
plt.xlabel('Numero de bins')
plt.ylabel('F1 score medio - 3 fold')
no_transf, = plt.plot(Ks, scores_1, 'b', label='sem transformacao')
equal_length, = plt.plot(Ks, scores_2, 'r', label='tamanhos iguais')
random_c, = plt.plot(Ks, fixed_05, 'k', label='random')

equal_frequency, = plt.plot(Ks, scores_3, 'g', label='frequencias iguais')
plt.legend([no_transf, equal_length, equal_frequency, random_c], ['sem transformacao', 'tamanhos iguais', 'frequencias iguais', 'random'])
plt.show()
