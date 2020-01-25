from numpy import exp, array, random, dot
from tools import Tools
import numpy as np
import math

class NeuralNetwork():


    def sigmoid(self, z):
        g = 1.0 / (1.0 + exp(-z))
        return g

    def sigmoidGradient(self, z):       
        #element-wise multiply     
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))        
        
    def randomInitializeWeights(self, L_in, L_out):
        epsilon_init = 0.12
        return random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init       
    
    def getCost(self, theta1, theta2, num_labels, X, y, reg_lambda):
        idmatrix = np.identity(num_labels)
        #y_matrix = idmatrix[y-1,:]    
        yy = y[:,0]    
        y_matrix = idmatrix[yy-1,:]        
        
        m = np.size(X, 0)         
        X = np.append(np.ones((m, 1)), X, axis=1)
        J = 0
        theta1_grad = np.zeros(np.shape(theta1))
        theta2_grad = np.zeros(np.shape(theta2))

        # Forward propagation ----------------------------------
        a1 = X
        
        z2 = a1.dot(theta1.conj().transpose())
        g1 = self.sigmoid(z2)        
        a2 = np.append(np.ones((np.size(g1, 0), 1)), g1, axis=1)
        z3 = a2.dot(theta2.conj().transpose())
        a3 = self.sigmoid(z3)
        h = a3

        # Unregularized cost function ----------------------------------
        #vector1 = np.multiply(-y_matrix, np.log(h))
        vector1 = (-y_matrix) * np.log(h)
        #vector2 = np.multiply((1 - y_matrix), (np.log(1 - h)))
        vector2 = (1 - y_matrix) * (np.log(1 - h))
        vec = vector1 - vector2        
        print('m: ', m)
                
        sm = vec.sum()
        
        #J_unregularized =  np.divide(sm, m)
        J_unregularized =  sm / m
        
                
        print('J_unregularized: ', J_unregularized)

        # Regularized cost function ----------------------------------

        # Remove first columns of theta1/theta2        
        t1 = np.delete(theta1, 0, 1)  
        t2 = np.delete(theta2, 0, 1)  
        regparam = (reg_lambda / (2 * m)) * (sum(sum(np.multiply(t1,t1))) + sum(sum(np.multiply(t2,t2))))
        print('regparam: ', regparam)
        # Add regularization param
        J = J_unregularized + regparam
        print('J: ', J)

        # Back propagation
        d3 = a3 - y_matrix
        d2 = np.multiply((d3 * t2), self.sigmoidGradient(z2))

        Delta1 = np.transpose(d2) * a1
        Delta2 = np.transpose(d3) * a2

        theta1_grad = (1/m) * Delta1
        theta2_grad = (1/m) * Delta2


        #T1 = theta1(:,1) = 0;
        #T2 = theta2(:,1) = 0;

        rp1 = (reg_lambda / m) * theta1
        rp2 = (reg_lambda / m) * theta2

        theta1_grad = theta1_grad + rp1
        theta2_grad = theta2_grad + rp2


        # Unroll gradients
        t = Tools()
        grad = t.unroll(theta1_grad, theta2_grad)
        # [Theta1_grad(:) ; Theta2_grad(:)];
        return J, grad