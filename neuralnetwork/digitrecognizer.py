import scipy.io
import numpy as np
from neuralnetwork import NeuralNetwork
from tools import Tools

# Setup the parameters 
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)



# Load Training Data
print('Loading and Visualizing Data ...\n')
mat = scipy.io.loadmat('a:\\ex4data1.mat')
X = mat['X']
m = np.shape(X)[0]
n = np.shape(X)[1]
print("Number of training examples: ", m)
print("Number of features: ", n)

y = mat['y']
print(np.shape(y))
#for x in range(4500, 5000):
  #print(y[x])


print('Loading Saved Neural Network Parameters ...')
tools = Tools()
# Load the weights into variables Theta1 and Theta2
theta = scipy.io.loadmat('a:\\ex4weights.mat')
theta1 = theta['Theta1']
theta2 = theta['Theta2']
#nn_params = tools.unroll(theta1, theta2)
reg_lambda = 1
nn = NeuralNetwork()
J = nn.getCost(theta1, theta2, num_labels, X, y, reg_lambda)


