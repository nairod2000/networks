import numpy as np

# need to look up important abstractions for a network to have. 
# ex: should layers be a flexible thing, should the input layer count as a layer, how to represent
# amount of neurons in each layer etc.

class Network:
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randint(-2, 3, (i, j)) for j, i in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randint(-2, 3, (i, 1)) for i in sizes[1:]]

    def forward(self, X):
        activations = list()
        axons = np.matmul(self.weights[0], X) + self.biases[0]
        activation = self.sigmoid(axons)
        activations.append(activation)
        for i in range(1, self.layers-1):
            axons = np.matmul(self.weights[i], activation) + self.biases[i]
            activation = self.sigmoid(axons)
            activations.append(activation)
        return activations

    def back_prop(self, neurons, expected):
        gradients = list()
        cost_to_L = self.cost_derivative(neurons[-1], expected)
        n = 0
        while n < len(self.weights):
            current_gradient = cost_to_L
            activation_to_z1 = self.sigmoid_prime(self.weights[-1] * neurons[-2])
            current_gradient *= activation_to_z1
            if n != 0:
                for i in range(n):
                    z_to_activation = self.weights[-1 - i]
                    activation_to_z = self.sigmoid_prime(self.weights[-2 - i] * neurons[-3 - i])
                    current_gradient *= (z_to_activation * activation_to_z)
            current_gradient *= neurons[-2 - n]
            n += 1
            gradients.append(current_gradient)
            if n == len(self.weights):
                break
        return gradients

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_prime(X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def cost_derivative(X, y):
        return (X - y)

    def train(self, lr, epocs):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    np.random.seed(1)
    network = Network([10, 4, 4, 3])
    X = np.random.randint(0, 4, (10, 1))
    print(network.forward(X))
