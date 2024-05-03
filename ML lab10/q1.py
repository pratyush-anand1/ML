import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, lr):
        self.weights = np.random.uniform(-0.3, 0.3, size=(input_size,))
        self.bias = np.random.uniform(-0.3, 0.3)
        self.lr = lr

    def threshold(self, x):
       if x < 0.5:
           return 0
       return 1

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.threshold(weighted_sum)

    def train(self, inputs, labels, epochs):
        accuracy_history = []

        for epoch in range(epochs):
            total_error = 0
            correct_predictions = 0

            for i in range(len(inputs)):
                input_data = inputs[i]
                label = labels[i]

                # Forward pass
                predicted = self.predict(input_data)

                # Compute the error
                error = label - predicted

                # Update weights and bias
                self.weights += self.lr * error * input_data
                self.bias += self.lr * error

                # Count correct predictions
                if abs(error) < 0.5:
                    correct_predictions += 1

                # Accumulate the total error for this epoch
                total_error += abs(error)

            # Calculate training accuracy for this epoch
            accuracy = correct_predictions / len(inputs)
            accuracy_history.append(accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}, Total Error: {total_error}")

        return accuracy_history

# Define the training data for OR and AND
or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_labels = np.array([0, 1, 1, 1])

and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_labels = np.array([0, 0, 0, 1])

# (a) Train the model for different learning rates and plot the training curve
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
epochs = 10

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    perceptron_or = Perceptron(input_size=2, lr=lr)
    accuracy_history_or = perceptron_or.train(or_inputs, or_labels, epochs)
    plt.plot(range(1, epochs + 1), accuracy_history_or, label=f'LR={lr}_OR')

    perceptron_and = Perceptron(input_size=2, lr=lr)
    accuracy_history_and = perceptron_and.train(and_inputs, and_labels, epochs)
    plt.plot(range(1, epochs + 1), accuracy_history_and, label=f'LR={lr}_AND')

plt.title('Training Curve for Different Learning Rates')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()
