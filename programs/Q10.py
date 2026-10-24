#10. Build a basic Multilayer Perceptron with one hidden layer to solve a classification problem. 

#Program 10: Build a basic Multilayer Perceptron with one hidden layer to solve a
#classification problem. 

import numpy as np

# 1. Define the Multilayer Perceptron Class
class MultilayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim=1, lr=0.1):
        """
        An MLP with 1 Hidden Layer for Binary Classification.
        """
        self.lr = lr
        
        # Initialize weights and biases randomly
        # W1 maps from input_dim to hidden_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        
        # W2 maps from hidden_dim to output_dim (1 neuron for binary classification)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, a):
        # derivative with respect to output activation 'a'
        return a * (1 - a)

    def forward(self, X):
        """Forward pass through the single hidden layer network."""
        # Hidden layer: Z1 = X * W1 + b1, A1 = sigmoid(Z1)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._sigmoid(self.Z1)
        
        # Output layer: Z2 = A1 * W2 + b2, A2 = sigmoid(Z2)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        """Backward propagation using gradient descent."""
        m = X.shape[0]
        
        # Output layer error (Binary Cross-Entropy derivative w.r.t Z2 simplifies to A2 - y)
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=10000):
        """Train the model over a specified number of epochs."""
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute Binary Cross Entropy Loss
            m = y.shape[0]
            loss = -np.sum(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15)) / m
            
            # Backward pass
            self.backward(X, y)
            
            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")

    def predict(self, X):
        """Predict binary classes (0 or 1) using a 0.5 decision threshold."""
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)


# 2. Generate a Non-Linearly Separable Dataset (XOR Problem)
# Linear models fail at XOR, making it a perfect benchmark for an MLP with 1 hidden layer.
X = np.array([[0, 0],[0, 1], [1, 0],[1, 1]])

y = np.array([[0], [1], [1], [0]])

# 3. Instantiate, Train, and Evaluate the Model
if __name__ == "__main__":
    print("--- Training Multilayer Perceptron on XOR ---")
    # 2 inputs, 4 neurons in the hidden layer, 1 output neuron
    mlp = MultilayerPerceptron(input_dim=2, hidden_dim=4, output_dim=1, lr=0.5)
    mlp.fit(X, y, epochs=5000)
    
    print("\n--- Predictions ---")
    predictions = mlp.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Target: {y[i][0]} -> Predicted: {predictions[i][0]}")
