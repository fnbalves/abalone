from util import *

from sklearn import tree

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_train, Y_train, X_test, Y_test] = pre_process_and_hold_out(X, Y)


accuracies = []
predictions = []

tree_classifier = tree.DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
tree_classifier.fit(X_train, Y_train)
print 'Score', tree_classifier.score(X_test, Y_test)#np.sqrt(mean_error(tree_classifier, X_test, Y_test))