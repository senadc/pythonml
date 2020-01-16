from numpy import exp, array, random, dot
import numpy as np
class NeuralNetwork():


    def sigmoid(self, z):
        g = 1.0 / (1.0 + exp(-z))
        return g

    def sigmoidGradient(self, z):            
        g = np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def randomInitializeWeights(self, L_in, L_out):
        epsilon_init = 0.12
        return random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
        