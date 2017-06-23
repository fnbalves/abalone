import numpy as np
from util import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings
from random_tansformations import *

featureset = load_features('fatures.pickle')

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process_and_hold_out(X, Y)

t_X_train = apply_features(X_train, featureset)
t_X_test = apply_features(X_test, featureset)

[mins, maxes] = get_max_and_mins(t_X_train)
st_X_train = scale_vars(t_X_train, mins, maxes)
st_X_test = scale_vars(t_X_test, mins, maxes)

scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []

Ks = []
for i in xrange(20):
    print i
    c = i + 1
    mlp_classifier_1 = MLPClassifier((c), activation='relu', solver='lbfgs', learning_rate='constant')
    mlp_classifier_2 = MLPClassifier((c), activation='relu', solver='lbfgs', learning_rate='constant')
    mlp_classifier_3 = MLPClassifier((c), activation='relu', solver='lbfgs', learning_rate='constant')
    mlp_classifier_4 = MLPClassifier((c), activation='relu', solver='lbfgs', learning_rate='constant')
    
    new_score_1 = np.mean(cross_val_score(mlp_classifier_1, X_f, Y_equal_frequency, cv=10))
    new_score_2 = np.mean(cross_val_score(mlp_classifier_2, X_f, Y_equal_frequency, cv=10))
    new_score_3 = np.mean(cross_val_score(mlp_classifier_3, X_f, Y_equal_frequency, cv=10))
    #new_score_4 = np.mean(cross_val_score(mlp_classifier_4, X_f, Y_equal_frequency, cv=10))
    
    Ks.append(c)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)
    #scores_4.append(new_score_4)
    
plt.xlabel('Tamanho da camada oculta')
plt.ylabel('Acuracias medias - 10 fold')
no_transf, = plt.plot(Ks, scores_1, 'b', label='sem transformacao')
equal_length, = plt.plot(Ks, scores_2, 'r', label='tamanhos iguais')
equal_frequency, = plt.plot(Ks, scores_3, 'g', label='frequencias iguais')
#random_transf, = plt.plot(Ks, scores_4, 'y', label='features aleatorias')

#plt.legend([no_transf, equal_length, equal_frequency, random_transf], ['sem transformacao', 'tamanhos iguais', 'frequencias iguais', 'features aleatorias'])
plt.legend([no_transf, equal_length, equal_frequency], ['sem transformacao', 'tamanhos iguais', 'frequencias iguais'])

plt.show()
