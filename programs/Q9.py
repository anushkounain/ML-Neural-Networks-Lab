#9. Use a single-layer perceptron to visualize decision boundaries on 2D data. 

#Program 9: Use a single-layer perceptron to visualize decision boundaries on 2D data. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 1. Define the Perceptron Class
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation_function(self, x):
        # Heaviside step function: return 1 if x >= 0, else 0
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Training loop
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Calculate linear output: z = w·x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Perceptron learning rule update if misclassified
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)


# 2. Generate Linearly Separable 2D Data
# We use 2 centers to create a binary classification problem
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=42)


# 3. Train the Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=50)
perceptron.fit(X, y)


# 4. Plotting the Data and Decision Boundary
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='o', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='s', label='Class 1')

# Calculate the decision boundary line
# Mathematical derivation: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_boundary = np.array([x1_min, x1_max])

w1, w2 = perceptron.weights
b = perceptron.bias

if w2 != 0:
    x2_boundary = -(w1 * x1_boundary + b) / w2
    plt.plot(x1_boundary, x2_boundary, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
else:
    # Fallback for vertical boundary line if w2 is exactly 0
    plt.axvline(x=-b/w1, color='black', linestyle='--', linewidth=2, label='Decision Boundary')

# Formatting the plot
plt.title("Single-Layer Perceptron Decision Boundary on 2D Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="upper left")
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(x1_min, x1_max)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

# Display the visualization
plt.show()
