from util import *

from sklearn import tree
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

accuracies = []
predictions = []

scores_1 = []
scores_2 = []
Ks = []
for i in xrange(100):
    print i+1, "of 100"
    tree_classifier_1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    tree_classifier_2 = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    Ks.append(i+1)
    
    scores_1.append(np.mean(cross_val_score(tree_classifier_1, X_f, Y_equal_frequency, cv=3, scoring='f1_macro')))
    scores_2.append(np.mean(cross_val_score(tree_classifier_2, X_f, Y_equal_frequency, cv=3, scoring='f1_macro')))

plt.xlabel('Profundidade maxima')
plt.ylabel('F1 score medio - 3 fold')

plt.xlabel('Valor de K')
plt.ylabel('Acuracia media - 3 fold')
entropy, = plt.plot(Ks, scores_1, 'b', label='entropy')
gini, = plt.plot(Ks, scores_2, 'r', label='gini')

plt.legend([entropy, gini], ['entropy', 'gini'])
plt.show()

uniform, = plt.plot(Ks, scores, 'b')

plt.show()

