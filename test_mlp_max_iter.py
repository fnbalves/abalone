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

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process_and_hold_out(X, Y)

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
    
    new_score_1 = np.mean(cross_val_score(mlp_classifier_1, X_f, Y_equal_frequency, cv=10))
    new_score_2 = np.mean(cross_val_score(mlp_classifier_2, X_f, Y_equal_frequency, cv=10))
    new_score_3 = np.mean(cross_val_score(mlp_classifier_3, X_f, Y_equal_frequency, cv=10))
    
    Ks.append(c)
    scores_1.append(new_score_1)
    scores_2.append(new_score_2)
    scores_3.append(new_score_3)
    
plt.xlabel('Tamanho da camada oculta')
plt.ylabel('Acuracia media - 10 fold')
iter_100, = plt.plot(Ks, scores_1, 'b', label='100 iter max')
iter_500, = plt.plot(Ks, scores_2, 'r', label='500 iter max')
iter_1000, = plt.plot(Ks, scores_3, 'g', label='1000 iter max')

plt.legend([iter_100, iter_500, iter_1000], ['100 iter max', '500 iter max', '1000 iter max'])
plt.show()
