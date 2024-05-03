import numpy as np
from sklearn.neural_network import MLPRegressor


network_architecture = [2, 2, 2]


activation_function = 'logistic'


model = MLPRegressor(hidden_layer_sizes=network_architecture, activation=activation_function, solver='sgd', learning_rate='adaptive')

input_hidden_weights = np.array([[0.1, 0.3],
                                [0.25, 0.8]])

hidden_output_weights = np.array([[0.2, 0.6],
                                 [0.4, 0.7]])


model.coefs_ = [input_hidden_weights, hidden_output_weights]
model.intercepts_ = [np.array([1, 1]), np.array([0, 0])]


X_train = np.array([[0.3, 0.8]])
y_train = np.array([[0.05, 0.6]])


model.fit(X_train, y_train)


X_pred = np.array([[0.5, 0.9]])


y_pred = model.predict(X_pred)


print("output -> " ,y_pred)
