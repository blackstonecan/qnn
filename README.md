
# QNN: Quantum-Inspired Neural Networks

**QNN** is a Python library designed to integrate quantum-inspired concepts into classical machine learning frameworks. It offers flexible, lightweight, and easy-to-use quantum-inspired layers, loss functions, and optimizers, making it ideal for researchers, students, and developers experimenting with quantum-inspired models.

---

## 🌟 Features

- **Quantum-Inspired Layers**: Layers using quantum-inspired probability calculations.
- **Custom Optimizers**: Includes `SGD` and `Adam` optimizers for training.
- **Modular Design**: Easily extendable with custom layers and models.
- **Cross-Entropy Loss**: Built-in support for cross-entropy loss.
- **Sequential API**: Simple and intuitive model building.

---

## 🚀 Quickstart

Here’s a minimal example to get started with QNN:

```python
import numpy as np
from qnn import Sequential, QuantumLayer, Adam

# Generate synthetic data
X = np.random.rand(100, 4)
y = np.random.randint(0, 3, size=100)  # 3-class classification

# Define the model
model = Sequential()
model.add(QuantumLayer(input_size=4, output_size=3, dropout_rate=0.1))
model.compile(loss='crossentropy', optimizer='adam', learning_rate=0.001)

# Train the model
model.fit(X, y, epochs=10, batch_size=16)

# Evaluate
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Predict
predictions = model.predict(X[:5])
print("Predictions:", predictions)
```

---

## 📜 Documentation

### Layers

- **`QuantumLayer`**: A quantum-inspired neural network layer.
  - **Parameters**:
    - `input_size` (int): Number of input features.
    - `output_size` (int): Number of output features (classes).
    - `dropout_rate` (float): Dropout rate during training (default: 0.0).
    - `l2_lambda` (float): L2 regularization coefficient (default: 0.01).

### Optimizers

- **`SGD`**: Stochastic Gradient Descent.
- **`Adam`**: Adaptive Moment Estimation.

### Loss Functions

- **`crossentropy_loss`**: Cross-entropy loss for classification tasks.

---

## 🤝 Acknowledgements

Inspired by quantum computing and machine learning principles.
