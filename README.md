# NeuralNetworkFromScracth
A fully connected neural network implemented from scratch using **NumPy**, without deep learning frameworks.  
Supports forward propagation, backpropagation, multiple activation functions, different loss functions, and L2 regularization.

---

## Features

- Custom forward and backward propagation
- L2 regularization (weight decay)
- Multiple activation functions:
  - ReLU
  - Sigmoid
  - Tanh
  - Linear
  - Softmax
- Multiple loss functions:
  - Cross-Entropy
  - Mean Squared Error (MSE)
- He and Xavier weight initialization
- Configurable learning rate
- Mini-batch training support

---

## Project Structure
├── neural_network.py
├── notebook.ipynb
└── README.md


---

## Initialization Methods

| Activation | Recommended Initialization |
|------------|---------------------------|
| ReLU       | He Initialization         |
| Sigmoid    | Xavier Initialization     |
| Tanh       | Xavier Initialization     |
| Softmax    | Xavier Initialization     |
| Linear     | Xavier Initialization     |

---

## Training Objective

The total loss optimized during training:

J = J_data + (λ / 2m) * Σ ||W||²

Where:
- `J_data` = Cross-Entropy or MSE
- `λ` = L2 regularization strength
- `m` = batch size

---

## Example Usage

```python
nn = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    activations=["relu", "relu", "softmax"],
    init_type="he"
)

nn.train(X_train, y_train, epochs=50, lr=0.01, lambda_=0.001)
```

Requirements

Python 3.9+

NumPy

Install dependencies:

```bash
pip install numpy
```

Future Improvements

Dropout regularization

Momentum / Adam optimizer

Batch normalization

Model saving/loading

Gradient checking

License

MIT License
