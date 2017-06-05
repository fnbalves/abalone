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
    mlp_classifier_1 = MLPClassifier((c), activation='identity', solver='lbfgs', learning_rate='constant')
    mlp_classifier_2 = MLPClassifier((c), activation='logistic', solver='lbfgs', learning_rate='constant')
    mlp_classifier_3 = MLPClassifier((c), activation='tanh', solver='lbfgs', learning_rate='constant')
    mlp_classifier_4 = MLPClassifier((c), activation='relu', solver='lbfgs', learning_rate='constant')
    mlp_classifier_1.fit(X_train, Y_train_3)
    mlp_classifier_2.fit(X_train, Y_train_3)
    mlp_classifier_3.fit(X_train, Y_train_3)
    mlp_classifier_4.fit(X_train, Y_train_3)
    
    predictions_1 = mlp_classifier_1.predict(X_test)
    predictions_2 = mlp_classifier_2.predict(X_test)
    predictions_3 = mlp_classifier_3.predict(X_test)
    predictions_4 = mlp_classifier_4.predict(X_test)
    
    new_score_1 = f1_score(Y_test_3, predictions_1, average='macro')
    new_score_2 = f1_score(Y_test_3, predictions_2, average='macro')
    new_score_3 = f1_score(Y_test_3, predictions_3, average='macro')
    new_score_4 = f1_score(Y_test_3, predictions_4, average='macro')
    
    Ks.append(c)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)
    scores_4.append(new_score_4)
    
plt.xlabel('Tamanho da camada oculta')
plt.ylabel('F1-score')
identity, = plt.plot(Ks, scores_1, 'b', label='identity')
logistic, = plt.plot(Ks, scores_2, 'r', label='logistic')
tanh, = plt.plot(Ks, scores_3, 'g', label='tanh')
relu, = plt.plot(Ks, scores_4, 'k', label='relu')

plt.legend([identity, logistic, tanh, relu], ['identity', 'logistic', 'tanh', 'relu'])
plt.show()
