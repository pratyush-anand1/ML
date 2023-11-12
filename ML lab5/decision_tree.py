import numpy as np
import pandas as pd

# Load the dataset (assuming you have it in a CSV file with no header)
# Manually specify column names and assign a default name for the target variable
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv("car_evaluation.csv", names=column_names)

# Define the dataset's attributes and target variable
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
target = 'class'  # Default name for the target variable

# Define a function to calculate entropy
def calculate_entropy(data):
    if len(data) == 0:
        return 0
    counts = data[target].value_counts()
    probabilities = counts / len(data)
    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy

# Define a function to split the dataset based on an attribute and value
def split_dataset(data, attribute, value):
    return data[data[attribute] == value]

# Define a function to find the best attribute to split on based on information gain
def find_best_split(data, attributes):
    entropy_parent = calculate_entropy(data)
    best_info_gain = 0
    best_attribute = None

    for attribute in attributes:
        unique_values = data[attribute].unique()
        entropy_children = 0

        for value in unique_values:
            subset = split_dataset(data, attribute, value)
            weight = len(subset) / len(data)
            entropy_children += weight * calculate_entropy(subset)

        info_gain = entropy_parent - entropy_children

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute = attribute

    return best_attribute

# Define the decision tree node structure
class TreeNode:
    def __init__(self, data, depth=0, max_depth=None):
        self.data = data
        self.depth = depth
        self.max_depth = max_depth

        if self.max_depth is not None and self.depth >= self.max_depth:
            self.label = data[target].mode().iloc[0]
        else:
            if len(data[target].unique()) == 1:
                self.label = data[target].iloc[0]
            elif len(attributes) == 0:
                self.label = data[target].mode().iloc[0]
            else:
                self.split_attribute = find_best_split(data, attributes)
                self.children = {}
                for value in data[self.split_attribute].unique():
                    child_data = split_dataset(data, self.split_attribute, value)
                    self.children[value] = TreeNode(child_data, depth=self.depth + 1, max_depth=self.max_depth)

# Build the decision tree
max_depth = 5  # Set your desired maximum depth
root = TreeNode(data, max_depth=max_depth)

# Define a function to make predictions using the decision tree
def predict(node, example):
    if hasattr(node, 'label'):
        return node.label
    else:
        value = example[node.split_attribute]
        if value in node.children:
            return predict(node.children[value], example)
        else:
            return data[target].mode().iloc[0]

# Load the test examples from the CSV file
test_examples = pd.read_csv("test_cases.csv")

# Make predictions for each test example
predictions = []
for _, example in test_examples.iterrows():
    prediction = predict(root, example)
    predictions.append(prediction)

# Display the predictions for the test examples
for i, prediction in enumerate(predictions):
    print(f"Test Example {i + 1}: Predicted class: {prediction}")
