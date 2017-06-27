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

[X_f, Y_no_transform, Y_equal_size, Y_equal_frequency] = pre_process(X, Y)

all_data = [[X_f[i], Y_equal_frequency[i]] for i, x in enumerate(X_f)]
random.shuffle(all_data)
len_train = int(2.0*float(len(all_data))/3.0)
train = all_data[:len_train]
test = all_data[len_train:]
X_train = [x for [x,y] in train]
Y_train = [y for [x,y] in train]
X_test = [x for [x,y] in test]
Y_test = [y for [x,y] in test]
    
mlp = MLPClassifier((10,), activation='logistic', max_iter=500)
mlp.fit(X_train, Y_train)

m = MLP((10,), activation='sigmoid', max_iter=10)
m.copy_sklearn(mlp)

predictions_mlp = mlp.predict(X_test)
predictions_m = m.predict(X_test)

score_mlp = accuracy_score(Y_test, predictions_mlp)
score_m = accuracy_score(Y_test, predictions_m)

print 'MLP', score_mlp
print 'My MLP', score_m

m.fit(X_f, Y_equal_frequency)
