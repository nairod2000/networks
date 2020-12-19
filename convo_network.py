import numpy as np 
from data_manager import MNISTLoader

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
# BUG not adding to history
# TODO Give documentation to everything
# TODO make a read me
# TODO: finish MaxPool
# TODO: Implement Conv2D

class Layer:

    differentiable_operations = 0
    operation_history = dict()
    gradients = dict()

    @classmethod
    def add_to_history(cls, obj, result, inpt):
        operation_number = obj.operation_number
        cls.operation_history[operation_number]['object'] = obj
        cls.operation_history[operation_number]['result'] = result
        cls.operation_history[operation_number]['input'] = inpt

    @classmethod
    def assign_order(cls, obj):
        cls.differentiable_operations += 1
        operation_number = cls.differentiable_operations
        cls.operation_history[operation_number] = dict()
        return operation_number

    @classmethod
    def clear_history(cls):
        cls.differentiable_operations = 0
        cls.operation_history = dict()
        cls.gradients = dict()

    @classmethod
    def add_to_gradients(cls):
        pass

    @staticmethod
    def is_row_vector(vector):
        pass

    @staticmethod
    def convert_to_row_vector(vector):
        pass

class Conv2D:
    '''
    Inputs
    - img_dim: needs to be tuple of (depth, height, width)
    - filter_d: needs to be a tuple of (depth, height, width)
    '''
    def __init__(self, in_dim, filter_d, channels, padding=0, stride=1):
        # Misc
        self.operation_number = None
        self.padding = padding
        self.stride = stride
        # Filter dimensions
        self.f_d, self.f_h, self.f_w = filter_d
        # Input dimensions
        self.in_d, self.in_h, self.in_w = in_dim
        # Output dimensions
        #self.out_w, self.out_h = self.output_dim((self.in_h, self.in_w))
        self.out_d = self.f_d
        # Weights and biases
        self.weight = np.random.rand(self.f_d, channels, self.f_h, self.f_w)
        self.bias = 0
        #self.bias = np.random.randn(self.f_d, 1, 1, 1)

    def __call__(self, X):
        # Calculate forward pass
        forward = self._convolve(X, self.weight)
        # Add the operation object and its result to history
        self.operation_number = Layer.assign_order(self)
        Layer.add_to_history(self, forward, X)
        return forward

    def derivative(self, grad, inpt):
        # Check dims of grad
        if self._is_row_vector(grad):
            grad = self._revert_from_row_vector(grad)
        # this is used by the optimizer object
        Layer.gradients[self.operation_number] = dict()
        self._conv_derivative_bias(grad) # Will do this one next
        self._conv_derivative_weight(grad, inpt)

    def _conv_derivative_weight(self, inpt, grad):
        '''
        The derivative with respect to the weight is a convolution of the input by the grad
        '''
        weight_grad = self._convolve(grad, inpt)
        Layer.gradients[self.operation_number]['weights'] = weight_grad

    def _conv_derivative_bias(self, grad):
        Layer.gradients[self.operation_number]['bias'] = 0

    def _convolve(self, inpt, convolver):
        '''
        Problem:
        - This only works for convolving the filter with the input.
        
        - It needs to be able to convolve the input by the gradient
        '''
        c_h, c_w = self.get_height_width(convolver)
        in_h, in_w = self.get_height_width(inpt)
        out_h, out_w = self.output_dim((in_h, in_w), (c_h, c_w))
        out_d = convolver.shape[0]
        # This needs to be calculated for each pass
        # self.f_h will need to be something like convolver.shape[-2]
        
        out = np.empty((out_d, out_h, out_w))
        
        for i in range(out_w):
            for j in range(out_h):
                # section of intrest
                soi = inpt[:, i:i+c_w, j:j+c_h]
                res = np.multiply(soi, convolver)
                sums = np.sum(res, axis=1)
                for k, arr in enumerate(sums):
                    out[k, j, i] = np.sum(arr)
        return out

    def output_dim(self, inpt_dim, conv_dim):
        height, width = inpt_dim
        c_height, c_width = conv_dim
        out_w = self._calculation(width, c_width)
        out_h = self._calculation(height, c_height)
        return int(out_w), int(out_h)

    def _calculation(self, inpt_dim, conv_dim):
        return ((inpt_dim + 2 * self.padding - conv_dim) / self.stride) + 1

    @staticmethod
    def get_height_width(X):
        dims = X.shape
        height, width = dims[-2], dims[-2]
        return height, width

    @staticmethod
    def _is_row_vector(vector):
        dims = len(vector.shape)
        if dims == 2:
            return True
        else:
            return False

    def _revert_from_row_vector(self, vect):
        vect = np.reshape(vect, (self.out_d, self.out_h, self.out_w))
        return vect


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
            

class Sigmoid:
    '''
    This is the sigmoid activation funciton.
    - __call__ is the forward function
    - derivative is used by the optimizer
    '''
    def __init__(self):
        self.operation_number = None # gets assigned when called

    def __call__(self, vector):
        result = self._sigmoid(vector)
        self.operation_number = Layer.assign_order(self)
        Layer.add_to_history(self, result, vector)
        return result

    def derivative(self, X):
        return self._sigmoid(X) * (1 - self._sigmoid(X))

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))


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


class Optimizer:
    '''
    Optimizer inherits the operation_history and uses the data to update the
    model. 
    - get_gradients gets the gradients for the necessary weights and biases
    - apply_gradients applies the gradients found in get_gradients
    '''

    def get_gradients(self, expected):
        num_operations = len(Layer.operation_history)
        
        # derivative from cost to activation
        last_layer_result = Layer.operation_history[num_operations]['result']
        cost_to_activation = self.cost_derivative_SGD(last_layer_result, expected.T)

        current_gradient = cost_to_activation

        #print(len(Layer.operation_history))
        for oper_idx in range(num_operations, 0, -2):
            # get necessary objects and relavent data
            non_lin_operation = Layer.operation_history[oper_idx]
            non_lin_object = non_lin_operation['object']
            lin_operation = Layer.operation_history[oper_idx-1]
            lin_object = lin_operation['object']
            #print(lin_object)

            ## derivative from activation to linear eq
            act_to_lin = non_lin_object.derivative(non_lin_operation['input'])

            # update gradient
            if act_to_lin.shape != current_gradient.shape:
                current_gradient = self.match_shapes(act_to_lin, current_gradient)
                #print(current_gradient.shape)
                #print(act_to_lin.shape)
            current_gradient *= act_to_lin

            # relavant gradients also stored in Layer.gradients, not returned
            lin_object.derivative(current_gradient, lin_operation['input'])

            if oper_idx != 2:
                # derivative from linear eq to prev activation
                prev_weight = lin_object.weight

                # update gradient
                current_gradient = np.matmul(prev_weight, current_gradient)


    def apply_gradients(self, lr):
        ''' 
        This is going to have 
        '''
        gradients = Layer.gradients
        operations = Layer.operation_history
        num_operations = len(Layer.operation_history) - 1
        for op in range(num_operations, 0, -2):
            # Get current objects and relavant data
            lin_object = operations[op]['object']
            weight_delta = gradients[op]['weights']
            bias_delta = gradients[op]['bias']

            # apply deltas
            if isinstance(lin_object, Conv2D):
                weight_delta = weight_delta[:, np.newaxis, :, :]
            lin_object.weight -= (lr * weight_delta)
            lin_object.bias -= (lr * bias_delta)

    @staticmethod
    def match_shapes(conv, lin):
        dims = conv.shape
        lin = np.reshape(lin, dims)
        return lin

    @staticmethod
    def cost_derivative_SGD(X, y):
        return (X - y)


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