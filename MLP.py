import numpy as np

class MLP(object):
    def __init__(self, hidden_layers_sizes, lambda_reg=0.0001, activation='sigmoid', learning_rate=0.1, min_increment=0.000000001, max_iter=500):
        self.hidden_layers_sizes = hidden_layers_sizes
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.min_increment = min_increment
        self.max_iter = max_iter
        self.cost_evolution = []
        self.activation = activation
        self.last_W = []
    
    def get_Y_set(self, Y):
        Y_set = []
        for y in Y:
            y_l = y.tolist()
            if y_l not in Y_set:
                Y_set.append(y_l)
        return Y_set

    def create_matrices(self, n_in, n_out):
        self.W = []
        self.last_W = []
        
        num_hidden_layers = len(self.hidden_layers_sizes)
        initial_size = n_in + 1
        for i in xrange(num_hidden_layers):
            new_W = 0.5*np.random.randn(initial_size, self.hidden_layers_sizes[i])
            initial_size = self.hidden_layers_sizes[i] + 1
            self.W.append(new_W)
            self.last_W.append(new_W)
        final_W = 0.5*np.random.randn(initial_size, n_out)
        self.W.append(final_W)

    def fit(self, X, Y):
        (num_elems, n_ins) = np.shape(X)
        self.available_labels = self.get_Y_set(Y)
        self.Y_train = self.convert_to_dummy(Y)
        
        self.X_train = X
        
        shape_Y = np.shape(self.Y_train)
        if len(shape_Y) == 1:
            n_out = 1
        else:
            n_out = shape_Y[1]

        self.create_matrices(n_ins, n_out)
        self.gradient_descend()

    def convert_to_dummy(self, Y):
        Y_dummy = []
        size_dummy = len(self.available_labels)
        
        for y in Y:
            y_l = y.tolist()
            new_Y = [0]*size_dummy
            ind = self.available_labels.index(y_l)
            new_Y[ind] = 1
            Y_dummy.append(new_Y)
            
        return np.array(Y_dummy)

    def act(self, X):
        if self.activation == 'relu':
            return self.relu(X)
        elif self.activation == 'tanh':
            return self.tanh(X)
        elif self.activation == 'identity':
            return self.identity(X)
        elif self.activation == 'sigmoid':
            return self.sigmoid(X)
        else:
            raise ValueError('Unknown activation ' + self.activation)

    def der_act(self, X):
        if self.activation == 'relu':
            return self.der_relu(X)
        elif self.activation == 'tanh':
            return self.der_tanh(X)
        elif self.activation == 'identity':
            return self.der_identity(X)
        elif self.activation == 'sigmoid':
            return self.der_sigmoid(X)
        else:
            raise ValueError('Unknown activation ' + self.activation)

    def relu(self, X):
        n_X = []
        (num_elems, dims) = np.shape(X)
        for i in xrange(num_elems):
            new_line = []
            for j in xrange(dims):
                new_line.append(max(X[i,j], 0))
            n_X.append(new_line)
        return np.asarray(n_X)

    def tanh(self, X):
        return np.tanh(X)

    def identity(self, X):
        return X

    def sigmoid(self, X):
        np_X = np.array(X)
        return 1/(1 + np.exp((-1)*np_X))

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

    def der_tanh(self, X):
        t = self.tanh(X)
        return 1 - t**2

    def der_identity(self, X):
        return np.ones(np.shape(X))

    def der_sigmoid(self, X):
        sig = self.sigmoid(X)
        return np.multiply(sig, 1 - sig)
    
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
            current_act = self.act(before_activation)
        return {'output' : current_act, 'activations' : activations, 'output_before' : before_activation}

    def forward(self, X):
        return self.forward_with_test_W(X, self.W)

    def cost_with_test_W(self, W):
        (m, d) = np.shape(self.X_train)
        m = float(m)
        Y_pred = np.reshape(self.forward_with_test_W(self.X_train, W)['output'], np.shape(self.Y_train))
        first_term = (-1)*np.sum(np.multiply(self.Y_train, np.log(Y_pred)) + np.multiply(1 - self.Y_train, np.log(1 - Y_pred)))#np.sum((self.Y_train - Y_pred)**2)
        reg_term = 0
        num_layers = len(W)
        for i in xrange(num_layers):
            reg_term += np.sum(W[i]**2)
        
        cost_val = (1.0/(1.0*m))*(first_term) + (self.lambda_reg/(2.0*m))*reg_term #1/2m
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
        out_raw = self.forward(X)['output']
        sm = self.softmax(out_raw)
        final_output = []
        for s in sm:
            final_output.append(self.available_labels[s])
        return final_output

    def softmax(self, Y):
        new_Y = []
        for y in Y:
            exp_y = np.exp(y)
            sum_y = np.sum(exp_y)
            sm = exp_y/sum_y
            new_Y.append(np.argmax(sm))
        return new_Y
    
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
            pred_der = np.ones(np.shape(np.transpose(self.der_act(pred_before))))
            activations = forw['activations']

            final_delta = np.multiply(np.transpose(pred - y), pred_der)
            last_delta = final_delta
            num_activations = len(activations)

            for j in xrange(num_activations - 1):
                curr_out = activations[num_activations - j - 1][0]
                curr_act_with_bias = activations[num_activations - j -1][1]

                Big_deltas[num_layers - j - 1] += np.dot(np.transpose(curr_act_with_bias), np.transpose(last_delta))

                der = np.transpose(self.der_act(curr_out))
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
            

    def make_W_with_sklearn_data(self, coefs_, intercepts_, classes_):
        num_matrices = len(coefs_)
        self.W = []
        for i in xrange(num_matrices):
            new_W = np.vstack((intercepts_[i], coefs_[i]))
            self.W.append(new_W)
        self.available_labels = classes_.tolist()

    def copy_sklearn(self, mlp):
        self.make_W_with_sklearn_data(mlp.coefs_, mlp.intercepts_, mlp.classes_)
        
    def gradient_descend(self):
        last_cost = self.cost()
        self.cost_evolution = [last_cost]

        has_exited_early = False
        
        for i in xrange(self.max_iter):
            print i
            self.one_step_grad_desc()
            next_cost = self.cost()
            self.cost_evolution.append(next_cost)
            
            cost_update = next_cost - last_cost
            
            if abs(cost_update) < self.min_increment:
                has_exited_early = True
                print 'Exited gradient descend by min increment'
                break
            last_cost = next_cost
        if not has_exited_early:
            print 'Exited gradient descend by max iterations'
