import random
from math import *
import pickle
import numpy as np

def create_random_feature():
    index_1 = 3 + floor(random.random()*7)
    index_2 = 3 + floor(random.random()*7)
    coef_1 = random.gauss(0,1)
    coef_2 = random.gauss(0,1)
    
    return [index_1, index_2, coef_1, coef_2]

def create_random_features():
    features = []
    for i in xrange(50):
        features.append(create_random_feature())
    return features

def save_features(features):
    out = open('fatures.pickle', 'wb')
    pickle.dump(features, out)
    out.close()

def load_features(path):
    return pickle.load(open(path, 'rb'))

def apply_features(X, features):
    (num_elems, dimensions) = np.shape(X)
    new_X = []
    for n in xrange(num_elems):
        x = X[n]
        nX = [x[0], x[1], x[2]]
        for f in features:
            new_f = pow(x[f[0]] + 0.001, f[2])*pow(x[f[1]] + 0.001, f[3])
            nX.append(new_f)
        new_X.append(nX)
    new_X_np = np.asarray(new_X)
    return new_X_np
