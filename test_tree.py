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

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process_and_hold_out(X, Y)

accuracies = []
predictions = []

tree_classifier = tree.DecisionTreeClassifier(criterion='entropy', class_weight='balanced')

print 'Score', np.mean(cross_val_score(tree_classifier, X_f, Y_equal_frequency, cv=10))
