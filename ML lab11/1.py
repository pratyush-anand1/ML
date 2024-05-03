import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


X = np.array([[0.3, 0.8]])

y = np.array([[0.05, 0.6]])


w1 = np.array([[0.1, 0.25], [0.3, 0.8]])

w2 = np.array([[0.2, 0.4], [0.6, 0.7]])


b1 = 1

b2 = 1


learning_rates =[0.5, 0.1 ,0.01]

for learning_rate in learning_rates:
 for i in range(10000):
   
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    y_hat = sigmoid(z2)

   
    error = y - y_hat

    d_y_hat = error * sigmoid_derivative(y_hat)

    error_hidden = d_y_hat.dot(w2.T)

    d_a1 = error_hidden * sigmoid_derivative(a1)

   
    w2 += a1.T.dot(d_y_hat) * learning_rate
    w1 += X.T.dot(d_a1) * learning_rate

    b2 += np.sum(d_y_hat, axis=0, keepdims=True) * learning_rate
    b1 += np.sum(d_a1, axis=0, keepdims=True) * learning_rate


 print("Output for learning rate ",learning_rate,"-> ",y_hat)
