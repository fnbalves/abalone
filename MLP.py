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

    def cost(self):
        (m, d) = np.shape(self.X_train)
        m = float(m)
        Y_pred = np.reshape(self.forward(self.X_train)[2], np.shape(self.Y_train))
        first_term = np.mean((self.Y_train - Y_pred)**2)
        reg_term = np.sum(self.W1**2) + np.sum(self.W2**2)
        cost_val = (-1.0/m)*(first_term) + (self.lambda_reg/(2.0*m))*reg_term
        return cost_val
    
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
        return [out_1, X_2, out_2]
    
    def backprop(self):
        Big_delta_1 = np.zeros(np.shape(self.W1))
        Big_delta_2 = np.zeros(np.shape(self.W2))
        (m, d) = np.shape(self.X_train)
        m = float(m)
        
        for i, x in enumerate(self.X_train):
            y = self.Y_train[i]
            outs = self.forward(x)
            a_1 = outs[0]
            outs_1 = outs[1]
            outs_2 = outs[2]
            
            delta_2 = y - outs_2
            delta_1 = np.multiply(self.der_relu(outs_1), np.dot(delta_2, np.transpose(self.W2)))   

            Big_delta_1 += np.dot(np.transpose(delta_1), a_1)
            Big_delta_2 += np.dot(np.transpose(delta_2), outs_2)

        reg_term_1 = np.copy(self.W1)
        reg_term_2 = np.copy(self.W2)
        reg_term_1[:,0] *= 0
        reg_term_2[:,0] *= 0

        Delta_1 = (1.0/m)*Big_delta_1 + self.lambda_reg*reg_term_1
        Delta_2 = (1.0/m)*Big_delta_2 + self.lambda_reg*reg_term_2

        return [Delta_1, Delta_2]

    def one_step_grad_desc(self):
        Deltas = self.backprop()
        self.W1 = self.W1 - 0.01*Deltas[0]
        self.W2 = self.W2 - 0.01*Deltas[1]

X = np.array([[1,2,3],[3,2,1]])
Y = np.array([1,2]).ravel()
m = MLP(3)
m.fit(X, Y)
