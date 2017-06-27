from util import *
from sklearn.neural_network import MLPClassifier
from MajorityVoteClassifier import *
from CombinedClassifier import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import time
import random

import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

scores_mlp = []
scores_mvc = []
scores_cc = []

Ts = []
for i in xrange(30):
    print i+1, 'of 30'
    Ts.append(i)
    
    [X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process(X, Y)

    all_data = [[X_f[i], Y_equal_frequency[i]] for i, x in enumerate(X_f)]
    random.shuffle(all_data)
    len_train = int(2.0*float(len(all_data))/3.0)
    train = all_data[:len_train]
    test = all_data[len_train:]
    X_train = [x for [x,y] in train]
    Y_train = [y for [x,y] in train]
    X_test = [x for [x,y] in test]
    Y_test = [y for [x,y] in test]
    
    mlp_classifier = MLPClassifier((10), activation='logistic', solver='lbfgs', learning_rate='constant', max_iter=500)
    mvc_classifier = MajorityVoteClassifier()
    cc_classifier = CombinedClassifier()
    
    mlp_classifier.fit(X_train, Y_train)
    mvc_classifier.fit(X_train, Y_train)
    cc_classifier.fit(X_train, Y_train)
    
    predictions_mlp = mlp_classifier.predict(X_test)
    predictions_mvc = mvc_classifier.predict(X_test)
    predictions_cc = cc_classifier.predict(X_test)
    
    new_score_mlp = f1_score(Y_test, predictions_mlp, average='macro')
    new_score_mvc = f1_score(Y_test, predictions_mvc, average='macro')
    new_score_cc = f1_score(Y_test, predictions_cc, average='macro')
    
    scores_mlp.append(new_score_mlp)
    scores_mvc.append(new_score_mvc)
    scores_cc.append(new_score_cc)
    
plt.xlabel('Tentativa')
plt.ylabel('F1 score')
mlp, = plt.plot(Ts, scores_mlp, 'b', label='mlp')
mvc, = plt.plot(Ts, scores_mvc, 'r', label='voto majoritario')
cc, = plt.plot(Ts, scores_cc, 'g', label='espaco de saida')
plt.legend([mlp, mvc, cc], ['mlp', 'voto majoritario', 'espaco de saida'])
plt.show()

