import pandas as pd
import numpy as np
import random

def custom_transform(X):
    (num_elems, dimensions) = np.shape(X)
    new_X = []
    for n in xrange(num_elems):
        x = X[n]
        nX = np.array([x[0], x[1], x[2], x[3], x[4], x[5], np.sqrt(x[6]), np.sqrt(x[7]), np.sqrt(x[8]), np.sqrt(x[9])])
        new_X.append(nX)
    new_X_np = np.asarray(new_X)
    return new_X_np


def separate_intervals(Y):
    dimensions = np.shape(Y)[0]
    new_Y = []
    for d in xrange(dimensions):
        nY = int(Y[d])/5
        new_Y.append(nY)
    return np.array(new_Y).ravel()

def read_file():
    raw_data = pd.read_csv("data/abalone.data", sep=",", header=None)
    return raw_data.values

def get_max_and_mins(X):
    (num_elems, dimensions) = np.shape(X)
    maxes = np.zeros((dimensions))
    mins = np.zeros((dimensions))
    for i in xrange(dimensions):
        mins[i] = 100000000
        maxes[i] = -10000000

    for i in xrange(num_elems):
        for d in xrange(dimensions):
            if X[i][d] < mins[d]:
                mins[d] = X[i][d]
            if X[i][d] > maxes[d]:
                maxes[d] = X[i][d]

    return [mins, maxes]

def scale_vars(X, mins, maxes):
    (num_elems, dimensions) = np.shape(X)
    n_X = []
    for i in xrange(num_elems):
        current_row = X[i]
        scaled_row = (current_row - mins)/(maxes - mins)
        n_X.append(scaled_row)
    return np.array(n_X)

def create_dummy_vars(X, index):
    (num_elems, dimensions) = np.shape(X)
    all_values = []
    for i in xrange(num_elems):
        all_values.append(X[i][index])
    all_values = list(set(all_values))
    len_all_values = len(all_values)
    n_X = []
    for i in xrange(num_elems):
        dummies = np.zeros((1, len_all_values))
        dummies[0][all_values.index(X[i][index])] = 1
        before = X[np.ix_([i], range(0, index))]
        after = X[np.ix_([i], range(index + 1, dimensions))]
        new_row = np.concatenate((before, dummies, after), axis=1)[0]
        n_X.append(new_row)
    return np.array(n_X)

def separate_X_Y(np_data):
    num_rows = np.shape(np_data)[0]
    num_colums = np.shape(np_data)[1]
    X = np_data[np.ix_(range(num_rows), range(num_colums-1))]
    Y  = np_data[np.ix_(range(num_rows), [num_colums - 1])]
    Y = Y.ravel()
    return [X, Y]

def pre_process_and_hold_out(X, Y_o):
    n_X = create_dummy_vars(X, 0)
    X_t = custom_transform(n_X)
    [mins, maxes] = get_max_and_mins(n_X)
    maxes[5] = 0.2
    n_X_scaled = scale_vars(X_t, mins, maxes)
    Y = separate_intervals(Y_o)
    
    all_data = [[x, Y[i]] for (i, x) in enumerate(n_X_scaled)]
    random.shuffle(all_data)
    len_data_set = len(all_data)
    train_set_size = 2*len_data_set/3
    train = [t for t in all_data[:train_set_size]]
    test = [t for t in all_data[train_set_size:]]

    X_train = np.array([x for [x,y] in train])
    Y_train = np.array([y for [x,y] in train])
    X_test = np.array([x for [x,y] in test])
    Y_test = np.array([y for [x,y] in test])

    return [X_train, Y_train, X_test, Y_test]

def mean_error(classifier, X_test, Y_test):
    predictions = classifier.predict(X_test)
    tam = float(np.shape(predictions)[0])
    error = predictions - Y_test
    error_t = np.transpose(error)
    error_q = np.dot(error, error_t)
    return error_q/tam

def isolate_feature(X, index):
    (num_elems, dimensions) = np.shape(X)
    feature = X[np.ix_(range(num_elems), [index])]
    return feature
