import numpy as np 
from typing import Callable

'''
flow:
- will create a Network object that will be composed of Layer objects
- will create an optimizer object that inherits Layer information
- will call forward on the model
- the Layer class will store necessary data to be optimized
- optimizer updates model using Layer data

- repeat
'''

# TODO: Implement back propagation
# TODO: Implement MaxPool
# TODO: Implement Conv2D

class Layer:

    differentiable_operations = 0
    operation_history = dict()
    gradients = dict()

    @classmethod
    def add_to_history(cls, obj, result):
        cls.differentiable_operations += 1
        operation_number = cls.differentiable_operations
        cls.operation_history[operation_number] = {'object': obj, 'result': result}

    @classmethod
    def add_to_gradients(cls):
        pass

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
        This subclass is for layers that will be fully connected to next layer.

        - will randomly initialize weights and biases.
        - __call__ is the forward function (for the user)
        - derivatives are used by the optimzer 
        '''
        def __init__(self, D_in, D_out):
            # initilize the weight and bias
            self.weight = np.random.randn(D_in, D_out)
            self.bias = np.random.randn(D_in, 1)

        def __call__(self, X):
            # Calculate forward pass
            forward = self._forward(X)
            # Add the operation object and its result to history 
            Layer.add_to_history(self, forward)

            return forward

        def derivative(self):
            # this is used by the optimizer object

            # this will call the other two derivative functions 
            pass

        def _linear_derivative_weight(self):
            # when this is called, it should automaticly add the result 
            # to a gradient dict
            pass

        def _linear_derivative_bias(self):
            # when this is called, it should automaticly add the result 
            # to a gradient dict
            pass

        def _forward(self, X):
            return np.multiply(self.weight, X) + self.bias
            

    class Sigmoid:
        '''
        This is the sigmoid activation funciton.

        - __call__ is the forward function
        - derivative is used by the optimizer
        '''
        def __call__(self, vector):
            result = self._sigmoid(vector)
            Layer.add_to_history(self, result)
            return result

        def derivative(self, X):
            return self._sigmoid(X) / (1 - self._sigmoid(X))

        @staticmethod
        def _sigmoid(X):
            return 1 / (1 + np.exp(X))


class Optimizer(Layer):
    '''
    Optimizer inherits the operation_history and uses the data to update the
    model. 

    - __call__ is to be called when the user whishes to update the model.
    '''
    def __init__(self):
        self.operations = Layer.operation_history
        self.gradients = Layer.gradients

    def get_gradients(self, expected):
        num_operations = len(self.operations)
        
        last_layer_result = self.operations[num_operations]['result']
        cost_to_activation = self.cost_derivative_SGD(last_layer_result, expected)

        current_derivative = cost_to_activation
        for oper_idx in range(num_operations, 0, -1):
            # this should just chain back and the important gradients will be stored 
            # automaticly. 
            pass

    def apply_gradients(self):
        # will use the self.gradiets to update the gradients of linear objects 
        # in Linear
        pass

    @staticmethod
    def cost_derivative_SGD(X, y):
        return (X - y)


class Network(Layer):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = self.Conv2D()
        self.layer2 = self.MaxPool()
        self.layer3 = self.Conv2D()
        self.layer4 = self.Sigmoid(self.Linear(784, 16))
        self.layer5 = self.Sigmoid(self.Linear(16, 16))

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.layer5(X)


if __name__ == '__main__':
    pass
