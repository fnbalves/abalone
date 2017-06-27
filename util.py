import pandas as pd
import numpy as np
import random
from random_tansformations import *

featureset = load_features('fatures.pickle')

def custom_transform(X):
    (num_elems, dimensions) = np.shape(X)
    new_X = []
    for n in xrange(num_elems):
        x = X[n]
        #nX =  np.array([x[0], x[1], x[2], x[3], x[4], x[5], pow(x[6], 0.02), pow(x[7], 0.02), pow(x[8], 0.02), pow(x[9], 0.02)])
        nX =  np.array([x[0]/(x[6] + 1), x[1]/(x[6] + 1), x[2]/(x[6] + 1), x[3]/(x[6] + 1), x[4]/(x[6] + 1), x[5]/(x[6] + 1), x[6]/(x[7] + 1), x[7]/(x[6] + 1), x[8]/(x[6] + 1), x[9]/(x[6] + 1)])
        new_X.append(nX)
    new_X_np = np.asarray(new_X)
    return new_X_np


def remove_outliers_from_5(X):
    for x in X:
        x[5] = min(0.25, x[5])

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

def pre_process(X, Y):
    n_X = create_dummy_vars(X, 0)
    remove_outliers_from_5(n_X)
    [mins, maxes] = get_max_and_mins(n_X)
    n_X_scaled = scale_vars(n_X, mins, maxes)

    Y_es = convert_Y_equal_size(Y)
    Y_ef = convert_Y_equal_frequency(Y)
    
        
    all_data = [[x, Y[i], Y_es[i], Y_ef[i]] for (i, x) in enumerate(n_X_scaled)]
    random.shuffle(all_data)

    X_final = np.array([x for [x, y1, y2, y3] in all_data], dtype=np.float32)
    Y_no_transform = np.array([y1 for [x, y1, y2, y3] in all_data], dtype=np.float32)
    Y_equal_size = np.array([y2 for [x, y1, y2, y3] in all_data], dtype=np.float32)
    Y_equal_frequency = np.array([y3 for [x, y1, y2, y3] in all_data], dtype=np.float32)
    
    return [X_final, Y_no_transform, Y_equal_size, Y_equal_frequency]

def pd_pre_process_and_hold_out(X, Y, num_bins):
    n_X = create_dummy_vars(X, 0)
    [mins, maxes] = get_max_and_mins(n_X)
    n_X_scaled = scale_vars(n_X, mins, maxes)

    Y_es = pd_equal_size(Y, num_bins)
    Y_ef = pd_equal_frequency(Y, num_bins)
    
        
    all_data = [[x, Y[i], Y_es[i], Y_ef[i]] for (i, x) in enumerate(n_X_scaled)]
    random.shuffle(all_data)

    X_final = np.array([x for [x, y1, y2, y3] in all_data], dtype=np.float32)
    Y_no_transform = np.array([y1 for [x, y1, y2, y3] in all_data], dtype=np.float32)
    Y_equal_size = np.array([y2 for [x, y1, y2, y3] in all_data], dtype=np.float32)
    Y_equal_frequency = np.array([y3 for [x, y1, y2, y3] in all_data], dtype=np.float32)
    
    return [X_final, Y_no_transform, Y_equal_size, Y_equal_frequency]

def test_custom_transformation(X, Y):
    n_X = create_dummy_vars(X, 0)
    X_t = apply_features(n_X, featureset)
    [mins_1, maxes_1] = get_max_and_mins(n_X)
    [mins_2, maxes_2] = get_max_and_mins(X_t)
    X_scaled_1 = scale_vars(n_X, mins_1, maxes_1)
    X_scaled_2 = scale_vars(X_t, mins_2, maxes_2)

    Y_es = convert_Y_equal_frequency(Y)

    return [X_scaled_1, X_scaled_2,  Y_es]

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

def convert_Y_equal_size(Y):
    n_Y = []
    for y in Y:
        if y < 10:
            n_Y.append(0)
        elif y < 20:
            n_Y.append(1)
        else:
            n_Y.append(2)
    return np.array(n_Y).ravel()

def convert_Y_equal_frequency(Y):
    n_Y = []
    for y in Y:
        if y < 9:
            n_Y.append(0)
        elif y < 11:
            n_Y.append(1)
        else:
            n_Y.append(2)
    return np.array(n_Y).ravel()

def pd_equal_size(Y, num_bins):
    return pd.cut(Y, num_bins, retbins=False, labels=range(num_bins))

def pd_equal_frequency(Y, num_bins):
    return pd.qcut(Y, num_bins, retbins=False, labels=range(num_bins))
