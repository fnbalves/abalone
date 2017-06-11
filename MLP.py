import numpy as np

class MLP(object):
    def __init__(self, hidden_layer_size=10, lambda_reg=1):
        self.hidden_layer_size = hidden_layer_size
        self.lambda_reg = lambda_reg
        
    def fit(self, X, Y):
        (num_elems, n_ins) = np.shape(X)

        self.X_train = X
        self.Y_train = Y
        
        shape_Y = np.shape(Y)
        if len(shape_Y) == 1:
            n_out = 1
        else:
            n_out = shape_Y[1]
        self.create_matrices(n_ins, n_out)
            
    def relu(self, X):
        n_X = []
        (num_elems, dims) = np.shape(X)
        for i in xrange(num_elems):
            new_line = []
            for j in xrange(dims):
                new_line.append(max(X[i,j], 0))
            n_X.append(new_line)
        return np.asarray(n_X)

    def der_relu(self, X):
        n_X = []
        (num_elems, dims) = np.shape(X)
        for i in xrange(num_elems):
            new_line = []
            for j in xrange(dims):
                if X[i,j] > 0:
                    new_line.append(1)
                else:
                    new_line.append(0)
            n_X.append(new_line)
        return np.asarray(n_X)
    
    def create_matrices(self, n_in, n_out):
        self.W1 = np.random.randn(n_in + 1, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size + 1, n_out)

    def forward(self, X):
        shape_X = np.shape(X)
        if len(shape_X) == 1:
            X = np.reshape(X, (1, len(X)))
        
        (num_elems, dims) = np.shape(X)
        bias = np.ones((num_elems, 1))
        X_1 = np.append(bias, X, axis=1)
        multi_1 = np.dot(X_1, self.W1)
        out_1 = self.relu(multi_1)
        X_2 = np.append(bias, out_1, axis=1)
        multi_2 = np.dot(X_2, self.W2)
        out_2 = self.relu(multi_2)
        return [X_2, out_2]
    
    def backprop(self):
        for i, x in enumerate(self.X_train):
            y = self.Y_train[i]
            outs = self.forward(x)
            outs_1 = outs[0]
            outs_2 = outs[1]
            
            delta_1 = self.Y_train - outs_2
            delta_2 = np.dot(self.der_relu(outs_1), np.dot(self.W2, delta_1))   
        
