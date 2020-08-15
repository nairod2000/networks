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
        axons = np.matmul(self.weights[0], X) + self.biases[0]
        activation = self.sigmoid(axons)
        for i in range(1, self.layers-1):
            axons = np.matmul(self.weights[i], activation) + self.biases[i]
            activation = self.sigmoid(axons)
        return activation

    def back_prop(self, prediction, real):
        pass

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    np.random.seed(1)
    network = Network([5, 4, 4])
    X = np.random.randint(0, 4, (5, 1))
    print(network.forward(X))