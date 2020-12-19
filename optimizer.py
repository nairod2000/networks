from layer import Layer
import numpy as np

class Optimizer(Layer):
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
            #if isinstance(lin_object, Conv2D):
            #    weight_delta = weight_delta[:, np.newaxis, :, :]
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

    @staticmethod
    def clear_history():
        Layer.clear_history()

    @staticmethod
    def assign_order(obj):
        Layer.assign_order(obj)

    @staticmethod
    def add_to_history(obj, result, inpt):
        Layer.add_to_history(obj, result, inpt)