import numpy as np 
from data_manager import MNISTLoader

from layer import Layer
from linear import Linear
from activations import Tanh
from conv2d import Conv2D
from optimizer import Optimizer

np.random.seed(1)
'''
flow:
- will create a Network object that will be composed of Layer objects
- will create an optimizer object that inherits Layer information
- will call forward on the model
- the Layer class will store necessary data to be optimized
- optimizer updates model using Layer data
- repeat
'''
# TODO Give documentation to everything
# TODO make a read me
# TODO: finish MaxPool
# TODO: Implement Conv2D

class Network:
    def __init__(self) -> None:
        super().__init__()
        self.activation = Tanh()

        #self.layer1 = self.Conv2D((1, 28, 28), (8, 3, 3), 1)
        #self.layer2 = self.MaxPool()
        #self.layer3 = self.Conv2D((2, 3, 3), 2)
        self.layer4 = Linear(784, 16)
        self.layer5 = Linear(16, 16)
        self.layer6 = Linear(16, 10)

    def forward(self, X):
        #X = self.activation(self.layer1(X))
        #X = self.layer2(X)
        #X = self.layer3(X)
        #X = X.flatten()[:, np.newaxis]
        X = self.activation(self.layer4(X))
        X = self.activation(self.layer5(X))
        X = self.activation(self.layer6(X))


if __name__ == '__main__':
    loader = MNISTLoader(64)
    epochs = 10
    X_train, y_train = loader.load_train()
    X_test, y_test = loader.load_test()
    learning_rate = 0.022

    model = Network()
    optimizer = Optimizer()

    for _ in range(epochs):
        for ex in range(len(X_train)):

            model.forward(X_train[ex].T / 100)

            optimizer.get_gradients(y_train[ex])

            optimizer.apply_gradients(learning_rate)
            Layer.clear_history()

    print('trained')
    correct = 0
    total = 0
    for ex in range(len(X_test)): # len(X_test)
        model.forward(X_test[ex].T)
        real = y_test[ex]
        pred = Layer.operation_history[6]['result'].T

        real = real.argmax(axis=1)
        pred = pred.argmax(axis=1)
        for i in range(len(real)):
            total += 1
            if real[i] == pred[i]:
                correct += 1

        Layer.clear_history()
    print(correct)
    print(total)
    print(correct/ total)