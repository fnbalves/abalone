import matplotlib.pyplot as plt
from util import *
from math import cos
import numpy as np
from random_tansformations import *

featureset = load_features('fatures.pickle')

remove_outliers = False

print 'Reading files'
np_data = read_file()
print 'Separate X and Y'
[X, Y] = separate_X_Y(np_data)
print 'Getting set of outputs'
all_outputs = sorted(set(Y.tolist()))

separated_Y = []
for o in all_outputs:
    new_Y = [y for y in Y if y == o]
    separated_Y.append(new_Y)


for (i, o) in enumerate(all_outputs):
    print 'Num of class', o,':' , len(separated_Y[i])
    
plt.xlabel('idade')
plt.ylabel('frequencia amostral')
plt.hist(Y)
plt.savefig('result_images/hist_ages.png')

n_X = create_dummy_vars(X, 0)

print 'Find outliers'
X_list = n_X.tolist()
sorted_X = sorted(X_list, reverse=True, key=lambda x:x[5])
print 'first outlier', sorted_X[0][5]
print 'second outlier', sorted_X[1][5]
print 'normal one', sorted_X[2][5]

if remove_outliers:
    remove_outliers_from_5(n_X)
[mins, maxes] = get_max_and_mins(n_X)
n_X_scaled = scale_vars(n_X, mins, maxes)

print 'Saving scatter plots'

(num_elems, dimensions) = np.shape(n_X)

names_features = ['dummy sexo M', 'dummy sexo F', 'dummy sexo I', 'comprimento', 'diametro', 'altura', 'peso inteiro', 'peso sugado', 'peso das visceras', 'peso da concha'] 
for d in xrange(dimensions):
    print 'feature', d + 1, 'of', dimensions
    feature = isolate_feature(n_X_scaled, d)
    plt.gcf().clear()
    plt.xlabel(names_features[d])
    plt.ylabel('idade (numero de aneis)')
    plt.scatter(feature, Y)
    plt.savefig('result_images/scatter_' + str(d) + '.png')
