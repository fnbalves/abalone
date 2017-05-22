import pandas as pd
import numpy as np
import random

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

def hold_out(X, Y):
    all_data = [[x, Y[i]] for (i, x) in enumerate(X)]
    random.shuffle(all_data)
    len_data_set = len(all_data)
    train_set_size = 2*len_data_set/3
    train = [t for t in all_data[:train_set_size]]
    test = [t for t in all_data[train_set_size:]]

    X_train = [x for [x,y] in train]
    Y_train = [y for [x,y] in train]
    X_test = [x for [x,y] in test]
    Y_test = [x for [x,y] in test]

    return [X_train, Y_train, X_test, Y_test]
