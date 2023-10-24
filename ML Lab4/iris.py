import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Hyperparameters
learning_rate = 0.01
num_epochs = 1000
num_folds = 5

# Shuffle the dataset
shuffle_indices = np.arange(len(X))
np.random.shuffle(shuffle_indices)
X_shuffled = X[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split the dataset into k folds
fold_size = len(X) // num_folds
fold_accuracies = []

for fold in range(num_folds):
    start_idx = fold * fold_size
    end_idx = (fold + 1) * fold_size
    
    # Prepare training and validation sets
    X_train = np.concatenate((X_shuffled[:start_idx], X_shuffled[end_idx:]))
    y_train = np.concatenate((y_shuffled[:start_idx], y_shuffled[end_idx:]))
    X_val = X_shuffled[start_idx:end_idx]
    y_val = y_shuffled[start_idx:end_idx]
    
    # Add a bias term to the input features
    X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_val_bias = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    
    # Convert labels to one-hot encoded vectors
    num_classes = len(np.unique(y_train))
    y_train_onehot = np.eye(num_classes)[y_train]
    
    # Initialize weights
    num_features = X_train_bias.shape[1]
    weights = np.random.randn(num_features, num_classes)
    
    # Training loop with gradient descent
    for epoch in range(num_epochs):
        # Forward pass
        logits = np.dot(X_train_bias, weights)
        probas = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Compute loss
        loss = -np.sum(y_train_onehot * np.log(probas)) / X_train_bias.shape[0]
        
        # Compute gradients
        gradient = np.dot(X_train_bias.T, (probas - y_train_onehot)) / X_train_bias.shape[0]
        
        # Update weights
        weights -= learning_rate * gradient
    
    # Make predictions on the validation fold
    logits_val_fold = np.dot(X_val_bias, weights)
    probas_val_fold = np.exp(logits_val_fold) / np.sum(np.exp(logits_val_fold), axis=1, keepdims=True)
    y_pred_fold = np.argmax(probas_val_fold, axis=1)
    
    # Calculate accuracy on the validation fold
    accuracy_fold = accuracy_score(y_val, y_pred_fold)
    fold_accuracies.append(accuracy_fold)

# Print accuracy for each fold
for i, accuracy in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy:", accuracy)

# Calculate average accuracy across all folds
average_accuracy = np.mean(fold_accuracies)
print("Average Accuracy:", average_accuracy)
