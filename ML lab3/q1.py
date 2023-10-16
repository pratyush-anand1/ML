import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x_train, y_train, iterations, learning_rate):
    m, num_features = x_train.shape
    coefficients = np.zeros(num_features)
    
    for i in range(iterations):
        z = np.dot(x_train, coefficients)
        pred = sigmoid(z)
        
        gradient = np.dot(x_train.T, (pred - y_train)) / m
        coefficients -= learning_rate * gradient

    return coefficients

dataset = pd.read_csv('diabetes_prediction_dataset.csv')
x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 4])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

alpha = 0.1
iterations = 100

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train logistic regression model
fin_coeff = logistic_regression(x_train, y_train, iterations, alpha)

# Calculate predictions on the test set using trained coefficients
z_test = np.dot(x_test, fin_coeff)
y_pred = sigmoid(z_test)

y_pred = (y_pred >= 0.5).astype(int)
pred_comp = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

print('Predicted VS Actual')

print(pred_comp[:30])

 