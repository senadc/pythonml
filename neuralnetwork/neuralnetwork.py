from numpy import exp, array, random, dot
import numpy as np
import math
import scipy

class NeuralNetwork(object):

    def __init__(self, input_layer_size, hidden_layer_size, num_labels):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.num_labels = num_labels

    def sigmoid(self, z):
        g = 1.0 / (1.0 + exp(-z))
        return g

    def sigmoid_gradient(self, z):       
        #element-wise multiply     
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))        
        
    def random_initialize_weights(self, L_in, L_out):
        epsilon_init = 0.12
        return random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init       
        
    def expand_y(self, y, num_labels):
        idmatrix = np.identity(num_labels)        
        y_onedim = y[:,0]    
        return idmatrix[y_onedim-1,:]   

    def unroll(self, a, b):
        a = np.vstack(a.reshape(a.size,order='F'))
        b = np.vstack(b.reshape(b.size,order='F'))
        return np.concatenate((a, b), axis=0)

    def reshape_weights(self, nn_params):
        theta1 = np.reshape(nn_params[:self.hidden_layer_size * (self.input_layer_size + 1)], 
                (self.hidden_layer_size, (self.input_layer_size + 1)), order="F")
        theta2 = np.reshape(nn_params[((self.hidden_layer_size * (self.input_layer_size + 1))):], 
                (self.num_labels, (self.hidden_layer_size + 1)), order="F")
        return theta1, theta2        

    def predict(self, theta1, theta2, X):
        m = np.size(X, 0)
        Xtmp = np.append(np.ones((m, 1)), X, axis=1)
        h1 = self.sigmoid(Xtmp.dot(theta1.transpose()))
        h1tmp = np.append(np.ones((m, 1)), h1, axis=1)
        h2 = self.sigmoid(h1tmp.dot(theta2.transpose()))
        return np.argmax(h2, 1) + 1          

    def get_nn_params(self, nn_params, X, y, reg_lambda):         
        thetas = self.reshape_weights(nn_params)
        theta1 = thetas[0]
        theta2 = thetas[1]
         
        y_matrix = self.expand_y(y, self.num_labels)        
        m = np.size(X, 0)         
        X = np.append(np.ones((m, 1)), X, axis=1)
        J = 0
        theta1_grad = np.zeros(np.shape(theta1))
        theta2_grad = np.zeros(np.shape(theta2))

        # Forward propagation ----------------------------------
        a1 = X        
        z2 = a1.dot(theta1.transpose())
        g1 = self.sigmoid(z2)        
        a2 = np.append(np.ones((np.size(g1, 0), 1)), g1, axis=1)
        z3 = a2.dot(theta2.transpose())
        a3 = self.sigmoid(z3)
        h = a3
        
        # Unregularized cost function ----------------------------------        
        vector1 = (-y_matrix) * np.log(h)        
        vector2 = (1 - y_matrix) * (np.log(1 - h))
        vec = vector1 - vector2                                
        sm = vec.sum()                
        J_unregularized =  sm / m
                       
        # Regularized cost function ----------------------------------
        # Remove first columns of theta1/theta2        
        t1 = np.delete(theta1, 0, 1)  
        t2 = np.delete(theta2, 0, 1)  
        regparam = (reg_lambda / (2 * m)) * (sum(sum(np.multiply(t1,t1))) + sum(sum(np.multiply(t2,t2))))

        # Add regularization param
        J = J_unregularized + regparam


        # Back propagation
        d3 = a3 - y_matrix
        d2 = d3.dot(t2) * self.sigmoid_gradient(z2)

        Delta1 = np.transpose(d2).dot(a1)
        Delta2 = np.transpose(d3).dot(a2)

        theta1_grad = (1/m) * Delta1
        theta2_grad = (1/m) * Delta2

        rp1 = (reg_lambda / m) * theta1
        rp2 = (reg_lambda / m) * theta2

        theta1_grad = theta1_grad + rp1
        theta2_grad = theta2_grad + rp2


        # Unroll gradients
        grad = self.unroll(theta1_grad, theta2_grad)
        return J, grad