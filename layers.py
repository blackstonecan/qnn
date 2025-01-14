import numpy as np

class Layer:
    def forward(self, X, training=True):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError


class QuantumLayer(Layer):
    def __init__(self, input_size, output_size, dropout_rate=0.0, l2_lambda=0.01):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, X, training=True):
        self.input = X
        self.raw_activations = X @ self.weights + self.biases
        self.amplitudes = np.tanh(self.raw_activations)

        if training and self.dropout_rate > 0.0:
            self.dropout_mask = (np.random.rand(*self.amplitudes.shape) > self.dropout_rate)
            self.amplitudes *= self.dropout_mask
        else:
            self.dropout_mask = 1
        
        squared_amps = self.amplitudes ** 2
        self.probabilities = squared_amps / np.sum(squared_amps, axis=1, keepdims=True)
        return self.probabilities

    def backward(self, grad_output):
        grad_amplitudes = grad_output * 2.0 * self.amplitudes / np.sum(self.amplitudes ** 2, axis=1, keepdims=True)
        grad_amplitudes *= self.dropout_mask
        grad_raw = grad_amplitudes * (1.0 - np.tanh(self.raw_activations) ** 2)
        self.grad_weights = self.input.T @ grad_raw + self.l2_lambda * self.weights
        self.grad_biases = np.sum(grad_raw, axis=0)
        grad_input = grad_raw @ self.weights.T
        return grad_input
