import numpy as np

"""
A perceptron takes in binary inputs. Each input has an asociated weight to it. The sum products of the binary
values multiplied by their corresponding weights is fed into the perceptron. The perceptron has a threshold value.
If the result is above the threshold it is a 1 else a 0.
"""

X = np.array([1, 0, 1])
weights = np.arange(3, 0, -1)
weights = np.reshape(weights, (len(weights), 1))
#print(np.dot(X, weights))

def perceptron(inpt, weight, threshold):
    score = np.dot(inpt, weight,)
    if score > threshold:
        return 1
    else:
        return 0

def sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

## Network of perceptrons
# In a linear algebra network, you have the weights, biases and the input. The input is a vector of activation
# scores. The weights is a matrix where the rows represents the amount of neurons in the next layer and the 
# columns in each row represent the conections of every neuron in the previous layer to each neuron (row) in the 
# next layer. The process of matrix multiplication, addition of the biases (vector) and activation function on the
# result is the next layer of neurons. It will be another vector of activation scores.

np.random.seed(1)
X = np.random.randint(0, 2, (20, 1)) # inputs (first layer neurons)
#print(X)
weights = np.random.uniform(-3, 3, (10, 20)) # weights connecting to all 10 neurons in next layer
#print(weights)
print(sigmoid(np.matmul(weights, X)))

# cost for network:
# (predicted - real)^2 + b 
