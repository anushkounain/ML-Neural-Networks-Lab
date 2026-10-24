#Program 6: Build and train a basic perceptron for binary classification including simple AND, OR functions

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        """
        Initialize the Perceptron.
        """
        # Initialize weights with zeros (or small random numbers)
        # We add 1 to the input size to account for the bias weight (w0)
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        """
        Heaviside unit step function.
        Returns 1 if x >= 0, else 0.
        """
        return 1 if x >= 0 else 0

    def predict(self, x):
        """
        Predict the class label for a single input vector.
        """
        # Combine bias and inputs: dot product of weights[1:] and x, plus weights[0] (bias)
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation_function(z)

    def train(self, X, y):
        """
        Train the weights using the Perceptron learning rule.
        """
        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                # 1. Forward pass: Calculate prediction
                prediction = self.predict(xi)
                
                # 2. Calculate error
                error = target - prediction
                
                # 3. Update weights if there's a misclassification
                if error != 0:
                    self.weights[1:] += self.lr * error * xi  # Update feature weights
                    self.weights[0] += self.lr * error        # Update bias weight
                    total_error += abs(error)
            
            # Optional: Stop early if the dataset is perfectly classified
            if total_error == 0:
                print(f"Converged early at epoch {epoch + 1}")
                break


# ==========================================
# Testing the Perceptron on AND & OR Gates
# ==========================================

# Standard 2-input binary dataset
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

# Target outputs
y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])

def evaluate_gate(gate_name, X, y):
    print(f"\n--- Training Perceptron for {gate_name} Gate ---")
    
    # Instantiate and train the model
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=20)
    perceptron.train(X, y)
    
    # Print the learned parameters
    print(f"Learned Bias: {perceptron.weights[0]:.2f}")
    print(f"Learned Weights: {perceptron.weights[1:]}")
    
    # Test predictions
    print("Testing Predictions:")
    for xi, target in zip(X, y):
        pred = perceptron.predict(xi)
        print(f"  Input: {xi} -> Target: {target} | Predicted: {pred}")

# Run evaluations
evaluate_gate("AND", X, y_and)
evaluate_gate("OR", X, y_or)
