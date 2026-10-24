#Program 8: Create a single-layer perceptron to classify a simple dataset (e.g., XOR problem).

import numpy as np

# 1. Setup reproducible seed and non-linear activation function
np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Define the XOR Dataset
# Inputs: [X1, X2]
X = np.array([[0, 0], [0, 1],[1, 0],[1, 1]])

# Target outputs for XOR
y = np.array([[0],[1],[1],[0]])

# 3. Initialize Neural Network Architecture (Topology: 2 Input -> 2 Hidden -> 1 Output)
epochs = 10000
learning_rate = 0.1

# Weights and biases initialization
weights_hidden = np.random.uniform(size=(2, 2))
bias_hidden = np.random.uniform(size=(1, 2))

weights_output = np.random.uniform(size=(2, 1))
bias_output = np.random.uniform(size=(1, 1))

# 4. Training Loop (Forward and Backward Propagation)
for epoch in range(epochs):
    # --- Forward Propagation ---
    hidden_layer_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # --- Backpropagation ---
    # Calculate error at output layer
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Calculate error at hidden layer
    error_hidden_layer = d_predicted_output.dot(weights_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases using gradient descent
    weights_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    weights_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# 5. Output Results
print("--- Final Trained Performance ---")
print("Target Output:\n", y)
print("\nRaw Model Predictions:\n", predicted_output)
print("\nRounded Binary Predictions:\n", np.round(predicted_output).astype(int))
