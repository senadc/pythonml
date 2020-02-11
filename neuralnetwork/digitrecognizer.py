import os
import scipy.io
import scipy.optimize
import numpy as np
from neuralnetwork import NeuralNetwork

def cost(nn_params, num_labels, X, y, input_layer_size, hidden_layer_size, reg_lambda):
    nn = NeuralNetwork(input_layer_size, hidden_layer_size, num_labels)
    return nn.get_nn_params(nn_params, X, y, reg_lambda)[0]
def gradient(nn_params, num_labels, X, y, input_layer_size, hidden_layer_size, reg_lambda):
    nn = NeuralNetwork(input_layer_size, hidden_layer_size, num_labels)
    return nn.get_nn_params(nn_params, X, y, reg_lambda)[1].flatten()


# Setup the parameters 
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)

dir_path = os.path.dirname(os.path.realpath(__file__))

print("Digit Recognizer -------------------------------\n")
# Load Training Data
print("Loading training data ...\n")
mat = scipy.io.loadmat(dir_path + '\\ex4data1.mat')
X = mat['X']
y = mat['y']
m = np.shape(X)[0]
n = np.shape(X)[1]
print("Number of training examples: ", m)
print("Number of features: ", n, "\n")


nn = NeuralNetwork(input_layer_size, hidden_layer_size, num_labels)
reg_lambda = 1

# Testing cost calculation...
print("Loading Saved Neural Network Parameters ...")
# Load the weights into variables Theta1 and Theta2
theta = scipy.io.loadmat(dir_path + '\\ex4weights.mat')
theta1 = theta['Theta1']
theta2 = theta['Theta2']
nn_params = nn.unroll(theta1, theta2)
result = nn.get_nn_params(nn_params, X, y, reg_lambda)
cost_value = round(result[0],4)
if cost_value != 0.3838:
    print("Test with saved parameters failed! Cost value: ", cost_value, "\n")


print("Training neural network. Please wait...")
initial_theta1 = nn.random_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = nn.random_initialize_weights(hidden_layer_size, num_labels)
initial_nn_params = nn.unroll(initial_theta1, initial_theta2)
weights = scipy.optimize.fmin_cg(cost, fprime=gradient, x0=initial_nn_params, 
        args=(num_labels, X, y, input_layer_size, hidden_layer_size, reg_lambda), 
        maxiter = 100, disp = False)
print("Training finished!\n")

print("Predicting...")

w1 = np.reshape(weights[:hidden_layer_size * (input_layer_size + 1)], 
    (hidden_layer_size, (input_layer_size + 1)), order="F")
w2 = np.reshape(weights[((hidden_layer_size * (input_layer_size + 1))):], 
    (num_labels, (hidden_layer_size + 1)), order="F")
        
predictions = nn.predict(w1, w2, X)

accuracy = np.mean(predictions == (y[:,0])) * 100

print("Training set accuracy: ", round(accuracy,2), "%")



