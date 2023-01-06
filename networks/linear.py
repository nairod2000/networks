from layer import Layer
import numpy as np

class Linear(Layer):
    '''
    This subclass is for layers that will be fully connected to next layer.
    - will randomly initialize weights and biases.
    - __call__ is the forward function (for the user)
    - derivatives are used by the optimzer 
    '''
    def __init__(self, D_in, D_out):
        # initilize the weight and bias
        self.weight = np.random.randn(D_in, D_out)
        self.bias = np.random.randn(D_out, 1)
        self.operation_number = None

    def __call__(self, vector):
        # Calculate forward pass
        result = self._forward(vector)
        # Add the operation object and its result to history
        self.operation_number = Layer.assign_order(self) 
        Layer.add_to_history(self, result, vector)
        return result

    def derivative(self, grad, inpt):
        # this is used by the optimizer object
        Layer.gradients[self.operation_number] = dict()
        self._linear_derivative_bias(grad)
        self._linear_derivative_weight(grad, inpt)

    def _linear_derivative_weight(self, grad, inpt):
        weight_grad = np.matmul(inpt, grad.T)
        Layer.gradients[self.operation_number]['weights'] = weight_grad

    def _linear_derivative_bias(self, grad):
        grad = np.sum(grad, axis=1)
        grad = grad[:, np.newaxis]
        Layer.gradients[self.operation_number]['bias'] = grad

    def _forward(self, X):
        result = np.matmul(self.weight.T, X) + self.bias
        return result

    @staticmethod
    def clear_history():
        Layer.clear_history()

    @staticmethod
    def assign_order(obj):
        Layer.assign_order(obj)

    @staticmethod
    def add_to_history(obj, result, inpt):
        Layer.add_to_history(obj, result, inpt)