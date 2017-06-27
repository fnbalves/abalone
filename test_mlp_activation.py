from util import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
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

scores_1 = []
scores_2 = []
scores_3 = []
scores_4 = []
Ks = []

for i in xrange(20):
    c = i + 1
    print c, 'of 20'
    mlp_classifier_1 = MLPClassifier((c), activation='identity', solver='lbfgs', learning_rate='constant')
    mlp_classifier_2 = MLPClassifier((c), activation='logistic', solver='lbfgs', learning_rate='constant')
    mlp_classifier_3 = MLPClassifier((c), activation='tanh', solver='lbfgs', learning_rate='constant')
    mlp_classifier_4 = MLPClassifier((c), activation='relu', solver='lbfgs', learning_rate='constant')
    
    new_score_1 = np.mean(cross_val_score(mlp_classifier_1, X_f, Y_equal_frequency, cv=3))
    new_score_2 = np.mean(cross_val_score(mlp_classifier_2, X_f, Y_equal_frequency, cv=3))
    new_score_3 = np.mean(cross_val_score(mlp_classifier_3, X_f, Y_equal_frequency, cv=3))
    new_score_4 = np.mean(cross_val_score(mlp_classifier_4, X_f, Y_equal_frequency, cv=3))
    
    Ks.append(c)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)
    scores_4.append(new_score_4)
    
plt.xlabel('Tamanho da camada oculta')
plt.ylabel('Acuracias medias - 3 fold')
identity, = plt.plot(Ks, scores_1, 'b', label='identity')
logistic, = plt.plot(Ks, scores_2, 'r', label='logistic')
tanh, = plt.plot(Ks, scores_3, 'g', label='tanh')
relu, = plt.plot(Ks, scores_4, 'k', label='relu')

plt.legend([identity, logistic, tanh, relu], ['identity', 'logistic', 'tanh', 'relu'])
plt.show()
