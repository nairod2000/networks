import numpy as np

# need to look up important abstractions for a network to have. 
# ex: should layers be a flexible thing, should the input layer count as a layer, how to represent
# amount of neurons in each layer etc.

class Network:
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randint(-2, 3, (j, i)) for j, i in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randint(-2, 3, (i, 1)) for i in sizes[1:]]

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
        gradients = list()
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
            current_gradient = np.matmul(current_gradient, neurons[-2 - n].T)
            n += 1
            gradients.append(current_gradient)
        return gradients

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_prime(self,X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    @staticmethod
    def cost_derivative(X, y):
        return (X - y)

    def train(self, lr, epocs):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    np.random.seed(1)
    network = Network([10, 5, 4, 3])
    X = np.random.randint(0, 4, (10, 1))
    activations = network.forward(X)
    print(len(activations))
    #print('neurons')
    #[print(x.shape) for x in activations]
    #print('--------')
    print('weights')
    [print(x.shape) for x in network.weights]
    #print('--------', '\n', 'backprop notes')
    deltas = network.back_prop(activations, [1, 0, 0])
    print('-------')
    print('deltas')
    [print(x.shape) for x in deltas]
    #print(len(deltas))
