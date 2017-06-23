import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from util import *
from MLP import *

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process_and_hold_out(X, Y)

mlp = MLPClassifier((10,), activation='logistic')
mlp.fit(X_f, Y_equal_frequency)

m = MLP((10,), activation='sigmoid')
m.copy_sklearn(mlp)

predictions_mlp = mlp.predict(X_f)
predictions_m = m.predict(X_f)

score_mlp = accuracy_score(Y_equal_frequency, predictions_mlp)
score_m = accuracy_score(Y_equal_frequency, predictions_m)

print 'MLP', score_mlp
print 'My MLP', score_m

m.fit(X_f, Y_equal_frequency)
print 'Numeric gradients', m.numeric_gradients()
print 'Backprop', m.backprop()

predictions_m = m.predict(X_f)
score_m = accuracy_score(Y_equal_frequency, predictions_m)
print 'My new MLP', score_m
