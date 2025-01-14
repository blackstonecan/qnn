import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.grad_weights
            if hasattr(layer, 'biases'):
                layer.biases -= self.learning_rate * layer.grad_biases


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        self.t = 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if not hasattr(layer, 'weights'):
                continue

            if i >= len(self.m_weights):
                self.m_weights.append(np.zeros_like(layer.weights))
                self.v_weights.append(np.zeros_like(layer.weights))
                self.m_biases.append(np.zeros_like(layer.biases))
                self.v_biases.append(np.zeros_like(layer.biases))

            m_w, v_w = self.m_weights[i], self.v_weights[i]
            m_b, v_b = self.m_biases[i], self.v_biases[i]

            m_w[:] = self.beta1 * m_w + (1 - self.beta1) * layer.grad_weights
            v_w[:] = self.beta2 * v_w + (1 - self.beta2) * (layer.grad_weights ** 2)
            m_b[:] = self.beta1 * m_b + (1 - self.beta1) * layer.grad_biases
            v_b[:] = self.beta2 * v_b + (1 - self.beta2) * (layer.grad_biases ** 2)

            m_w_hat = m_w / (1 - self.beta1 ** self.t)
            v_w_hat = v_w / (1 - self.beta2 ** self.t)
            m_b_hat = m_b / (1 - self.beta1 ** self.t)
            v_b_hat = v_b / (1 - self.beta2 ** self.t)

            layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            layer.biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
