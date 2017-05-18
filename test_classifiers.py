import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

def read_file():
    raw_data = pd.read_csv("data/abalone.data", sep=",", header=None)
    return raw_data.values

def separate_X_Y(np_data):
    num_rows = np.shape(np_data)[0]
    num_colums = np.shape(np_data)[1]
    X = np_data[np.ix_(range(num_rows), range(num_colums-1))]
    Y  = np_data[np.ix_(range(num_rows), [num_colums - 1])]
    Y = Y.ravel()
    return [X, Y]

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
