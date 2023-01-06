from layer import Layer
import numpy as np

class Sigmoid(Layer):
    '''
    This is the sigmoid activation funciton.
    - __call__ is the forward function
    - derivative is used by the optimizer
    '''
    def __init__(self):
        self.operation_number = None # gets assigned when called

    def __call__(self, vector):
        result = self._sigmoid(vector)
        self.operation_number = self.assign_order(self)
        self.add_to_history(self, result, vector)
        return result

    def derivative(self, X):
        return self._sigmoid(X) * (1 - self._sigmoid(X))

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def assign_order(obj):
        Layer.assign_order(obj)

    @staticmethod
    def add_to_history(obj, result, inpt):
        Layer.add_to_history(obj, result, inpt)


class Tanh:
    def __init__(self):
        self.operation_number = None # Gets assigned when called

    def __call__(self, vector):
        result = self._tanh(vector)
        self.operation_number = Layer.assign_order(self)
        Layer.add_to_history(self, result, vector)
        return result

    def derivative(self, X):
        return 1.0 - (np.square(self._tanh(X)))

    @staticmethod
    def _tanh(X):
        return np.tanh(X)

    @staticmethod
    def assign_order(obj):
        Layer.assign_order(obj)

    @staticmethod
    def add_to_history(obj, result, inpt):
        Layer.add_to_history(obj, result, inpt)