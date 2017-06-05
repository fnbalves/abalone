from util import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train_1, Y_train_2, Y_train_3, X_test, Y_test_1, Y_test_2, Y_test_3] = pre_process_and_hold_out(X, Y)

scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []
Ks = []
for i in xrange(20):
    c = i + 1
    mlp_classifier_1 = MLPClassifier((c), activation='tanh', solver='lbfgs', learning_rate='invscaling', max_iter=100)
    mlp_classifier_2 = MLPClassifier((c), activation='tanh', solver='lbfgs', learning_rate='invscaling', max_iter=500)
    mlp_classifier_3 = MLPClassifier((c), activation='tanh', solver='lbfgs', learning_rate='invscaling', max_iter=1000)
    mlp_classifier_1.fit(X_train, Y_train_3)
    mlp_classifier_2.fit(X_train, Y_train_3)
    mlp_classifier_3.fit(X_train, Y_train_3)
    
    predictions_1 = mlp_classifier_1.predict(X_test)
    predictions_2 = mlp_classifier_2.predict(X_test)
    predictions_3 = mlp_classifier_3.predict(X_test)
    
    new_score_1 = f1_score(Y_test_3, predictions_1, average='macro')
    new_score_2 = f1_score(Y_test_3, predictions_2, average='macro')
    new_score_3 = f1_score(Y_test_3, predictions_3, average='macro')
    
    Ks.append(c)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)
    
plt.xlabel('Tamanho da camada oculta')
plt.ylabel('F1-score')
iter_100, = plt.plot(Ks, scores_1, 'b', label='100 iter max')
iter_500, = plt.plot(Ks, scores_2, 'r', label='500 iter max')
iter_1000, = plt.plot(Ks, scores_3, 'g', label='1000 iter max')

plt.legend([iter_100, iter_500, iter_1000], ['100 iter max', '500 iter max', '1000 iter max'])
plt.show()
