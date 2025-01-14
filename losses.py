import numpy as np

def crossentropy_loss(y_true, y_pred, epsilon=1e-9):
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

def crossentropy_grad(y_true, y_pred, epsilon=1e-9):
    return -y_true / (y_pred + epsilon) / y_true.shape[0]

def one_hot_encode(labels, num_classes):
    """
    Convert integer labels of shape (N,) to one-hot of shape (N, num_classes).
    """
    return np.eye(num_classes)[labels]