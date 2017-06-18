import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

class MLP(object):
    def __init__(self, hidden_layers_sizes, lambda_reg=1, learning_rate=0.0001, power_T=0.5, max_fail=10, min_increment=0.000000001, max_iter=500):
        self.hidden_layers_sizes = hidden_layers_sizes
        self.lambda_reg = lambda_reg
        self.power_T = power_T
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.max_fail = max_fail
        self.min_increment = min_increment
        self.max_iter = max_iter
        self.cost_evolution = []

    def create_matrices(self, n_in, n_out):
        self.W = []
        num_hidden_layers = len(self.hidden_layers_sizes)
        initial_size = n_in + 1
        for i in xrange(num_hidden_layers):
            new_W = np.random.randn(initial_size, self.hidden_layers_sizes[i])
            initial_size = self.hidden_layers_sizes[i] + 1
            self.W.append(new_W)
        final_W = np.random.randn(initial_size, n_out)
        self.W.append(final_W)

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

    def forward_with_test_W(self, X, W):
        shape_X = np.shape(X)
        if len(shape_X) == 1:
            X = np.reshape(X, (1, len(X)))

        activations = []
        current_act = X
        before_activation = X
        num_layers = len(W)
        for i in xrange(num_layers):
            (num_elems, dims) = np.shape(current_act)
            bias = np.ones((num_elems, 1), dtype=np.float32)
            act_with_bias = np.append(bias, current_act, axis=1)
            activations.append((before_activation, act_with_bias))
            before_activation = np.dot(act_with_bias, W[i])
            current_act = self.relu(before_activation)
        return {'output' : current_act, 'activations' : activations, 'output_before' : before_activation}

    def forward(self, X):
        return self.forward_with_test_W(X, self.W)

    def cost_with_test_W(self, W):
        (m, d) = np.shape(self.X_train)
        m = float(m)
        Y_pred = np.reshape(self.forward_with_test_W(self.X_train, W)['output'], np.shape(self.Y_train))
        first_term = np.sum((self.Y_train - Y_pred)**2)
        reg_term = 0
        num_layers = len(W)
        for i in xrange(num_layers):
            reg_term += np.sum(W[i]**2)
        
        cost_val = (1.0/(2.0*m))*(first_term) + (self.lambda_reg/(2.0*m))*reg_term
        return cost_val

    def cost(self):
        return self.cost_with_test_W(self.W)
    
    def numeric_gradients(self):
        Num_Deltas = []
        num_layers = len(self.W)
        test_W = []
        for i in xrange(num_layers):
            test_W.append(np.copy(self.W[i]))
        
        for i in xrange(num_layers):
            new_grad = np.zeros(np.shape(self.W[i]))
            pert_grad = np.zeros(np.shape(self.W[i]))
            p = 0.0001
            (ne,d) = np.shape(new_grad)
            for l in xrange(ne):
                for c in xrange(d):
                    pert_grad[l, c] = p
                    
                    backup_W = np.copy(test_W[i])
                    
                    test_W[i] += pert_grad
                    cost1 = self.cost_with_test_W(test_W)
                    
                    test_W[i] = np.copy(backup_W)
                    test_W[i] -= pert_grad
                    cost2 = self.cost_with_test_W(test_W)

                    grad = (cost1 - cost2)/(2*p)
                    new_grad[l,c] = grad

                    pert_grad[l,c] = 0
                    test_W[i] = np.copy(backup_W)
                    
            Num_Deltas.append(new_grad)

        return Num_Deltas
    
    def predict(self, X):
        return self.forward(X)['output']
        
    def backprop(self):
        Big_deltas = []
        num_layers = len(self.W)

        for i in xrange(num_layers):
            Big_deltas.append(np.zeros(np.shape(self.W[i])))
            
        (m, d) = np.shape(self.X_train)
        m = float(m)
        
        for i, x in enumerate(self.X_train):
            y = self.Y_train[i]
            forw = self.forward(x)
            pred = forw['output']
            pred_before = forw['output_before']
            pred_der = np.transpose(self.der_relu(pred_before))
            activations = forw['activations']

            final_delta = np.multiply(np.transpose(pred - y), pred_der)
            last_delta = final_delta
            num_activations = len(activations)

            for j in xrange(num_activations - 1):
                curr_out = activations[num_activations - j - 1][0]
                curr_act_with_bias = activations[num_activations - j -1][1]

                Big_deltas[num_layers - j - 1] += np.dot(np.transpose(curr_act_with_bias), np.transpose(last_delta))

                der = np.transpose(self.der_relu(curr_out))
                small_W = self.W[num_layers - j -1][1:, :]
                
                new_delta = np.multiply(np.dot(small_W, last_delta), der)

                last_delta = new_delta
            curr_act_with_bias = activations[0][1]
            Big_deltas[0] += np.dot(np.transpose(curr_act_with_bias), np.transpose(last_delta))

        for i in xrange(num_layers):
            reg_term = np.copy(self.W[i])
            reg_term[:,0] *= 0
            Big_deltas[i] += self.lambda_reg*reg_term
            Big_deltas[i] *= (1.0/m)

        return Big_deltas

    def one_step_grad_desc(self):
        Deltas = self.backprop()
        num_layers = len(self.W)
        for i in xrange(num_layers):
            self.W[i] -= self.learning_rate*Deltas[i]
        self.learning_rate*=self.power_T

    def gradient_descend(self):
        last_cost = self.cost()
        self.learning_rate = self.initial_learning_rate
        num_failure = 0
        self.cost_evolution = [last_cost]

        has_exited_early = False
        
        for i in xrange(self.max_iter):
            self.one_step_grad_desc()
            next_cost = self.cost()
            self.cost_evolution.append(next_cost)
            
            cost_update = next_cost - last_cost
            if cost_update > 0:
                num_failure += 1
                if num_failure > self.max_fail:
                    print 'Exited gradient descend by max_failure'
                    has_exited_early = True
                    break

            if abs(cost_update) < self.min_increment:
                has_exited_early = True
                print 'Exited gradient descend by min increment'
                break
            last_cost = next_cost
        if not has_exited_early:
            print 'Exited gradient descend by max iterations'

#X = np.array([[0,0], [0,1], [1,1], [1,0]]*100 + 100*[[1,1]], dtype=np.float32)
#Y = np.array([0,0,1,0]*100 + [1]*100, dtype=np.float32)
X = np.array([[1,2,3],[3,2,1]], dtype=np.float32)
Y = np.array([[1,2], [2,1]], dtype=np.float32)
#mlp = MLPClassifier((3,), activation='relu', solver='lbfgs', learning_rate='invscaling')
#mlp.fit(X, Y)
m = MLP((3,), power_T=1, lambda_reg=0)
m.fit(X, Y)
#m.gradient_descend()
