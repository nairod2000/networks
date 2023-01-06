class Layer:

    differentiable_operations = 0
    operation_history = dict()
    gradients = dict()

    @classmethod
    def add_to_history(cls, obj, result, inpt):
        operation_number = obj.operation_number
        #print(operation_number)
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