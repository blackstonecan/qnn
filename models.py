import numpy as np

from .optimizers import SGD, Adam
from .losses import crossentropy_loss, crossentropy_grad, one_hot_encode
from .layers import Layer

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.loss_grad_fn = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss='crossentropy', optimizer='sgd', learning_rate=0.01):
        if loss == 'crossentropy':
            self.loss_fn = crossentropy_loss
            self.loss_grad_fn = crossentropy_grad
        else:
            raise ValueError(f"Unknown loss function: {loss}")
        
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training=training)
        return X
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def fit(self, X, y, epochs=10, batch_size=32, shuffle=True):
        n_samples = X.shape[0]
        
        if len(y.shape) == 1:
            num_classes = len(np.unique(y))
            y = one_hot_encode(y, num_classes)
        
        for epoch in range(epochs):
            if shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X, y = X[indices], y[indices]

            batch_losses = []
            for i in range(0, n_samples, batch_size):
                X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
                y_pred = self.forward(X_batch, training=True)
                loss = self.loss_fn(y_batch, y_pred)
                batch_losses.append(loss)

                grad = self.loss_grad_fn(y_batch, y_pred)
                self.backward(grad)
                self.optimizer.step(self.layers)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(batch_losses):.4f}")
            
    def evaluate(self, X, y):
        if len(y.shape) == 1:
            y_int = y
            num_classes = len(np.unique(y))
            y = one_hot_encode(y, num_classes)
        else:
            # If user already passed in one-hot labels,
            # we need an integer copy for accuracy calculation
            y_int = np.argmax(y, axis=1)
        
        y_pred = self.forward(X, training=False)
        loss = self.loss_fn(y, y_pred)
        y_pred_int = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_int == y_pred_int)
        return loss, accuracy
    
    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)
