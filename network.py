import numpy as np
from load_data import *


class Network: 
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.array(np.random.randint(-2, 3, (j, i)), dtype='float64') for j, i in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.array(np.random.randint(-2, 3, (i, 1)), dtype='float64') for i in sizes[1:]]

    def forward(self, X):
        '''
        Takes the input to the network and feeds it forwards to the output layer
        --------------------------------------
        inputs:
        X:
        - the inputs to the network
        --------------------------------------
        return:
        - A list of each activation layer
        '''
        activations = list()
        activations.append(X)
        unactivated_layer = np.matmul(self.weights[0].T, X) + self.biases[0]
        activated_layer = self.sigmoid(unactivated_layer)
        activations.append(activated_layer)
        for i in range(1, self.layers-1):
            unactivated_layer = np.matmul(self.weights[i].T, activated_layer) + self.biases[i]
            activated_layer = self.sigmoid(unactivated_layer)
            activations.append(activated_layer)
        return activations

    def back_prop(self, neurons, expected):
        '''
        inputs:

        neurons:
        - a list of the all the activated neuron layers
        expected:
        - what the network was supposed to output
        X:
        - what the input for the sample was
        --------------------------------------
        return:
        - the gradients of all the layers weights
        '''
        expected = np.array(expected)
        expected_vector = np.reshape(expected, (expected.shape[0], 1))
        weights_delta = list()
        biases_delta = list()
        cost_to_L = self.cost_derivative(neurons[-1], expected_vector)
        n = 0
        while n < len(self.weights):
            current_gradient = cost_to_L
            activation_to_z1 = self.sigmoid_prime(np.matmul(self.weights[-1].T, neurons[-2]))
            current_gradient *= activation_to_z1
            if n != 0:
                for i in range(n):
                    z_to_activation = self.weights[-1 - i]
                    current_gradient = np.matmul(z_to_activation, current_gradient)
                    activation_to_z = self.sigmoid_prime(np.matmul(self.weights[-2 - i].T, neurons[-3 - i]))
                    current_gradient *= activation_to_z
            biases_delta.append(current_gradient)
            current_gradient = np.matmul(neurons[-2 - n], current_gradient.T)
            weights_delta.append(current_gradient)
            n += 1
        return list(reversed(weights_delta)), list(reversed(biases_delta))

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_prime(self,X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    @staticmethod
    def cost_derivative(X, y):
        return (X - y)

    def train(self, lr, examples, real_values, epochs):
        samples = len(real_values)
        for _ in range(epochs):
            for i in range(samples):
                neurons = self.forward(examples[i])
                weights_deltas, biases_deltas = self.back_prop(neurons, real_values[i])
                for j in range(len(weights_deltas)):
                    self.weights[j] -= lr * weights_deltas[j]
                    self.biases[j] -= lr * biases_deltas[j]

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]

    def score_model(self, X, y):
        # for each example, run forward, argmax on last activation
        # if argmax == lable correct += 1 else wrong += 1
        total_examples = len(X)
        correct = 0
        for i, example in enumerate(X):
            activations = self.forward(example)
            precieved_value = np.argmax(activations[-1])
            real_value = np.argmax(y[i])
            if precieved_value == real_value:
                correct += 1
        return correct / total_examples


if __name__ == '__main__':
    data = load_data()
    expected_vector = produce_expected_vectors(data)
    data = reshape_input_vector(data)
    network = Network([784, 16, 16, 10])
    network.train(.001, data, expected_vector, 10)


    test_data = load_test()
    expected_vector = produce_expected_vectors(test_data)
    test_data = reshape_input_vector(test_data)
    print(f'Model score {network.score_model(test_data, expected_vector)}')
