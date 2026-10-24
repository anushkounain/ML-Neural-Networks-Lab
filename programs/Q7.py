#Program 7: Implement and visualize different activation functions like: Step, Sigmoid,
# Tanh, and ReLU

#An activation function is a mathematical formula applied to a neuron's input in an
#Activation function to decide whether the neuron should fire and what output signal
#to pass to the next layer.

#1. The Binary Step function is a simple threshold-based function.
#If the input value is greater than or equal to a certain threshold (usually zero),
#the neuron fires and outputs a 1; otherwise, it outputs a 0


#2. The Sigmoid (or logistic) function squashes any real-valued input into an
# S-shaped curve between 0 and 1. This makes it highly useful for models
# where outputs need to be interpreted as probabilities

#3. The Tanh function is very similar to Sigmoid, but it squashes values into a
#range between -1 and 1. Because its output is zero-centered, negative inputs
#map to negative outputs and positive inputs map to positive outputs, which usually
#helps the network learn faster than Sigmoid

#4. ReLU is the most widely used activation function in deep learning today.
#It passes positive inputs directly through, but completely zeroes out any
#negative inputs

import numpy as np

def sigmoid(x):
    """Outputs values strictly between 0 and 1. Perfect for binary classification."""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Outputs values between -1 and 1. Zero-centered, ideal for hidden layers."""
    return np.tanh(x)

def relu(x):
    """Returns 0 for negative values, and the input itself for positive values."""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Prevents 'dying neurons' by keeping a tiny slope for negative inputs."""
    return np.where(x > 0, x, x * alpha)

def softmax(x):
    """Normalizes an array into probabilities that sum up exactly to 1."""
    exp_x = np.exp(x - np.max(x)) # Subtracting max prevents numerical overflow
    return exp_x / exp_x.sum(axis=0)


# Create sample input data (simulating neural network raw layer outputs)
sample_data = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])

print("Raw Input Array: ", sample_data)
print("-" * 50)
print("Sigmoid Output:  ", np.round(sigmoid(sample_data), 4))
print("Tanh Output:     ", np.round(tanh(sample_data), 4))
print("ReLU Output:     ", np.round(relu(sample_data), 4))
print("Leaky ReLU:      ", np.round(leaky_relu(sample_data), 4))
print("Softmax Output:  ", np.round(softmax(sample_data), 4))
print("Softmax Sum:     ", np.sum(softmax(sample_data))) # Will equal 1.0
