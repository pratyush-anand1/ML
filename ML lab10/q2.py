import numpy as np

# (a) Generate random data
np.random.seed(42)

num_samples = 1000
num_features = 5
num_classes = 2

# Generate random data with 5 features
X = np.random.rand(num_samples, num_features)

# Generate random labels (0 or 1)
y = np.random.randint(2, size=(num_samples, 1))

# (b) Implement a feedforward neural network from scratch
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_input)

        return self.predicted_output

    # (c) Implement forward pass and calculate cross entropy loss
    def calculate_loss(self, X, y):
        # Forward pass
        predicted_output = self.forward(X)

        # Calculate cross-entropy loss
        loss = -np.mean(y * np.log(predicted_output) + (1 - y) * np.log(1 - predicted_output))
        return loss

# Instantiate the neural network
input_size = num_features
hidden_size = 4  # You can adjust this based on your requirements
output_size = num_classes

neural_network = NeuralNetwork(input_size, hidden_size, output_size)

# Calculate the initial loss
initial_loss = neural_network.calculate_loss(X, y)
print(f"Initial Cross-Entropy Loss: {initial_loss}")
