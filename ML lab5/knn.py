import csv
import random
import math
import operator

# Load the dataset
def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append(row)
    return dataset

# Convert attribute values to numerical values
def convert_attributes(dataset):
    attribute_mapping = {
        'vhigh': 4, 'high': 3, 'med': 2, 'low': 1,
        '2': 2, '3': 3, '4': 4, '5more': 5,
        'more': 6, 'small': 1, 'med': 2, 'big': 3,
        'low': 1, 'med': 2, 'high': 3,
        'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4
    }

    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = attribute_mapping[row[i]]
    return dataset

# Split the dataset into training and test sets
def split_dataset(dataset, split_ratio):
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    return dataset[:split_index], dataset[split_index:]

# Calculate Euclidean distance between two instances
def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1) - 1):
        distance += (instance1[i] - instance2[i]) ** 2
    return math.sqrt(distance)

# Find k nearest neighbors
def get_neighbors(training_set, test_instance, k):
    distances = []
    for train_instance in training_set:
        dist = euclidean_distance(test_instance, train_instance)
        distances.append((train_instance, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = [distance[0] for distance in distances[:k]]
    return neighbors

# Make a prediction based on majority class of neighbors
def predict(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

# Calculate accuracy of predictions
def accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0

# Perform k-fold cross-validation
def cross_validation(dataset, k_folds, k_value):
    fold_size = len(dataset) // k_folds
    accuracy_scores = []

    for i in range(k_folds):
        start = i * fold_size
        end = (i + 1) * fold_size
        validation_set = dataset[start:end]
        training_set = dataset[:start] + dataset[end:]

        predictions = []
        for test_instance in validation_set:
            neighbors = get_neighbors(training_set, test_instance, k_value)
            result = predict(neighbors)
            predictions.append(result)

        accuracy_score = accuracy(validation_set, predictions)
        accuracy_scores.append(accuracy_score)
        print(f"Fold {i + 1} Accuracy: {accuracy_score:.2f}%")

    average_accuracy = sum(accuracy_scores) / k_folds
    print(f"Average Accuracy: {average_accuracy:.2f}%")

def main():
    filename = "car_evaluation.csv"
    dataset = load_dataset(filename)
    dataset = convert_attributes(dataset)

    k_folds = 5
    k_value = 5

    print(f"Performing {k_folds}-fold cross-validation with k={k_value}")
    cross_validation(dataset, k_folds, k_value)

if __name__ == "__main__":
    main()
