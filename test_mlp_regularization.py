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

scores = []

Ks = []

for i in xrange(200):
    print i+1, 'of 200'
    c = float(i+1)/100.0
    
    mlp_classifier = MLPClassifier((10), activation='logistic', solver='lbfgs', learning_rate='constant', alpha=c)
    
    new_score = np.mean(cross_val_score(mlp_classifier, X_f, Y_equal_frequency, cv=3))
    
    Ks.append(c)
    scores.append(new_score)
    
plt.xlabel('Regularization term')
plt.ylabel('Acuracias medias - 3 fold')

all_values, = plt.plot(Ks, scores, 'b')
plt.show()
