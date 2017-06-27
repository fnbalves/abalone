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

Ks = []

for i in xrange(10):
    c = i + 1
    print c, 'of 10'
    mlp_classifier_1 = MLPClassifier(tuple(c*[10]), activation='logistic', solver='lbfgs', learning_rate='constant')
    mlp_classifier_2 = MLPClassifier(tuple(c*[10]), activation='relu', solver='lbfgs', learning_rate='constant')
    
    new_score_1 = np.mean(cross_val_score(mlp_classifier_1, X_f, Y_equal_frequency, cv=3))
    new_score_2 = np.mean(cross_val_score(mlp_classifier_2, X_f, Y_equal_frequency, cv=3))
    
    Ks.append(c)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    
plt.xlabel('Numero de camadas ocultas')
plt.ylabel('Acuracias medias - 3 fold')

logistic, = plt.plot(Ks, scores_1, 'b', label='funcao logistica')
relu, = plt.plot(Ks, scores_2, 'r', label='relu')
plt.legend([logistic, relu], ['funcao logistica', 'relu'])
plt.show()
