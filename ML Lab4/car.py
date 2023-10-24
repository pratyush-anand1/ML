import csv
import random
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(float))

    def fit(self, dataset):
        # Calculate class probabilities
        total_instances = len(dataset)
        class_counts = defaultdict(int)

        for instance in dataset:
            class_label = instance[-1]
            class_counts[class_label] += 1

        for class_label, count in class_counts.items():
            self.class_probs[class_label] = count / total_instances

        # Calculate feature probabilities
        attribute_value_counts = defaultdict(lambda: defaultdict(int))

        for instance in dataset:
            class_label = instance[-1]
            for i, attr_value in enumerate(instance[:-1]):
                attribute_value_counts[class_label][i, attr_value] += 1

        for class_label, attribute_counts in attribute_value_counts.items():
            for (attr_index, attr_value), count in attribute_counts.items():
                self.feature_probs[class_label][attr_index, attr_value] = count / class_counts[class_label]

    def predict_instance(self, instance):
        max_prob = float('-inf')
        predicted_class = None

        for class_label in self.class_probs.keys():
            prob = self.class_probs[class_label]

            for i, attr_value in enumerate(instance[:-1]):
                prob *= self.feature_probs[class_label][i, attr_value]

            if prob > max_prob:
                max_prob = prob
                predicted_class = class_label

        return predicted_class

    def predict(self, dataset):
        predictions = [self.predict_instance(instance) for instance in dataset]
        return predictions

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append(row)
    return dataset

def convert_attributes_to_numeric(dataset):
    buying_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
    maint_mapping = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
    doors_mapping = {'5more': 6}
    persons_mapping = {'more': 7}
    lug_boot_mapping = {'small': 0, 'med': 1, 'big': 2}
    safety_mapping = {'low': 0, 'med': 1, 'high': 2}
    decision_mapping = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
    
    mapping_dicts = [buying_mapping, maint_mapping, doors_mapping, persons_mapping,
                     lug_boot_mapping, safety_mapping, decision_mapping]

    converted_dataset = []

    for instance in dataset:
        converted_instance = [mapping.get(attr, attr) for mapping, attr in zip(mapping_dicts, instance)]
        converted_dataset.append(converted_instance)
    
    return converted_dataset

def split_dataset(dataset, split_ratio):
    # Split the dataset into training and testing sets
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    training_set = dataset[:split_index]
    testing_set = dataset[split_index:]
    return training_set, testing_set

def get_accuracy(testing_set, predictions):
    correct_predictions = sum(1 for actual, predicted in zip(testing_set, predictions) if actual[-1] == predicted)
    total_predictions = len(testing_set)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def main():
    filename = 'car_evaluation.csv'  
    split_ratio = 0.8  # 80% training, 20% testing

    dataset = load_dataset(filename)
    dataset = convert_attributes_to_numeric(dataset)  # Convert string attributes to numeric
    training_set, testing_set = split_dataset(dataset, split_ratio)

    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(training_set)

    predictions = nb_classifier.predict(testing_set)
    accuracy = get_accuracy(testing_set, predictions)
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
