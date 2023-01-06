import os
import numpy as np
from pandas import read_csv

np.random.seed(0)

class MNISTLoader:
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_path = os.path.join('data', 'train.csv')
        self.test_path = os.path.join('data', 'mnist_test.csv')

    def load_train(self):
        
        # Load raw data
        df = read_csv(self.train_path)
        data = np.array(df)
        np.random.shuffle(data)
        y_real = data[:, 0]
        X = np.delete(data, 0, axis=1)

        # One hot encode y_real
        y_real = self._one_hot_output(y_real)
        
        # Split into batches
        X, y_real = self._create_batches(X, y_real)

        return X, y_real

    def load_test(self):
        df = read_csv(self.test_path)
        data = np.array(df)
        np.random.shuffle(data)
        y_real = data[:, 0]
        X = np.delete(data, 0, axis=1)

        y_real = self._one_hot_output(y_real)

        X, y_real = self._create_batches(X, y_real)

        return X, y_real

    def _create_batches(self, X, y):

        num_batches = np.ceil(X.shape[0] / self.batch_size).astype(int)
        X = np.array_split(X, num_batches)
        y = np.array_split(y, num_batches)

        return X, y
    
    def _one_hot_output(self, y):
        encoded_ys = list()
        for i in y:
            vector = np.zeros(10)
            vector[i] = 1
            encoded_ys.append(vector)
        return np.array(encoded_ys)


if __name__ == "__main__":
    loader = MNISTLoader(64)
    X, y = loader.load_train()
