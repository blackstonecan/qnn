"""
QNN: A quantum-inspired neural network library.
"""
__version__ = "0.1.0"

# Import key components for top-level access
from .layers import QuantumLayer
from .optimizers import SGD, Adam
from .models import Sequential
from .losses import crossentropy_loss, crossentropy_grad
