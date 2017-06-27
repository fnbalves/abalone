from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import EditedNearestNeighbours
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

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

all_data = [[X_f[i], Y_equal_frequency[i]] for i, x in enumerate(X_f)]
random.shuffle(all_data)
len_train = int(2.0*float(len(all_data))/3.0)
train = all_data[:len_train]
test = all_data[len_train:]
X_train = [x for [x,y] in train]
Y_train = [y for [x,y] in train]
X_test = [x for [x,y] in test]
Y_test = [y for [x,y] in test]
    
scores_1 = []
scores_2 = []

Ks = []

enn = EditedNearestNeighbours(ratio='all', n_neighbors=5)
X_enn, Y_enn = enn.fit_sample(X_train, Y_train)

for i in xrange(20):
    K = 2*i + 1
    print i+1, 'of 20'
    knn_classifier_1 = KNeighborsClassifier(n_neighbors=K)
    knn_classifier_2 = KNeighborsClassifier(n_neighbors=K)

    knn_classifier_1.fit(X_train, Y_train)
    knn_classifier_2.fit(X_enn, Y_enn)
    
    predictions_1 = knn_classifier_1.predict(X_test)
    predictions_2 = knn_classifier_2.predict(X_test)
    
    new_score_1 = f1_score(Y_test, predictions_1, average='macro')
    new_score_2 = f1_score(Y_test, predictions_2, average='macro')
    
    Ks.append(K)

    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    
plt.xlabel('Valor de K')
plt.ylabel('F1-score')
no_resample, = plt.plot(Ks, scores_1, 'b', label='sem edicao')
with_resample, = plt.plot(Ks, scores_2, 'r', label='com edicao')
plt.legend([no_resample, with_resample], ['sem edicao', 'com edicao'])
plt.show()
