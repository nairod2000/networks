import numpy as np 
from typing import Callable


'''
flow:
- will create a Network object that will be composed of Layer object
- will call forward on the model
- the model will store necessary data to be optimized
- will pass the model to the optimizer
- optimizer updates model

ToDo: Linear needs to contain what oporations were done on each forward pass.
'''

class Layer:
    '''
    Every layer object is expected to be a two dimensional tensor.
    '''

    class Conv2D:
        def __init__(self):
            pass

        def __call__(self, X):
            pass


    class MaxPool:
        def __init__(self):
            pass

        def __call__(self):
            pass


    class Linear:
        '''
        This subclass is for layers that will be fully connected. 
        Is just a feedforward network.

        - will randomly initialize weights and biases.
        - will need a history of the operations done on it.
        - __call__ is the forward function
        - should implement a backward function that uses the the history 
          to update weights and biases
        '''

        linear_layers = 0
        linear_data = dict()

        def __init__(self, D_in, D_out):
            # initilize the instance data
            self.gradients = list()
            self.weight = np.random.randn(D_in, D_out)
            self.bias = np.random.randn(D_in, 1)

            # Update class data
            self.linear_layers += 1
            self.linear_data[self.linear_layers] = {'weight': self.weight, 'bias': self.bias}

            self.id = self.linear_layers

        def __call__(self, X, direction='f'):
            if direction == 'f':
                forward = self._forward(X)
                self.linear_data[self.id]['forward'] = forward
                return forward
            elif direction == 'b':
                return self._linear_derivative()

        def _forward(self, X):
            return np.multiply(self.weight, X) + self.bias

        def _linear_derivative(self):
            pass
            

    class Activation:
        def __init__(self, activation_type):
            self.act_dict = {'sig': self._sigmoid}
            self.div_dict = {'sig': self._sigmoid_derivative}

            try:
                self.activation = self.act_dict[activation_type]
                self.derivative = self.div_dict[activation_type]
            except AttributeError:
                print(f'Activation has no attribute {activation_type}')


        def __call__(self, vector, direction='f'):
            if direction == 'f':
                return self.activation(vector) 
            elif direction == 'd':
                return self.derivative(vector)

        @staticmethod
        def _sigmoid(X):
            return 1 / (1 + np.exp(X))

        def _sigmoid_derivative(self, X):
            return self._sigmoid(X) / (1 - self._sigmoid)


class Optimizer:
    '''
    This should be passed the model object (obj of Network class)
    '''
    def __init__(self, model, activation, method):
        self.model = model
        self.model_activation = activation
        self.method = method

    def update(self, expected):
        pass

    @staticmethod
    def cost_derivative_SGD(X, y):
        return (X - y)


class Network(Layer):
    def __init__(self, activation: Callable[np.array, str]) -> None:
        super().__init__()

        self.layer1 = self.Conv2D()
        self.layer2 = self.MaxPool()
        self.layer3 = self.Conv2D()
        self.layer4 = activation(self.Linear(784, 16))
        self.layer5 = activation(self.Linear(16, 16))

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.layer5(X)


if __name__ == '__main__':
    pass
