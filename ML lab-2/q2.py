import numpy as np
import matplotlib.pyplot as plt

def calculate_cost(X, Y, slope, intercept):
    predictions = slope * X + intercept
    squared_errors = (predictions - Y) ** 2
    cost = np.mean(squared_errors) / 2
    return cost

def compute_gradient(x,y,w,b):
    training_examples = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(training_examples):
        f_wb = w*x[i] + b
        dj_dw_cur = (f_wb - y[i])*x[i]
        dj_db_cur = (f_wb - y[i])
        dj_dw = dj_dw + dj_dw_cur
        dj_db = dj_db + dj_db_cur
    dj_dw = dj_dw/training_examples
    dj_db = dj_db/training_examples
    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,aplha,iterations,compute_gradient):

    w = w_in
    b = b_in

    for i in range(iterations):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w = w - aplha*dj_dw
        b = b - aplha*dj_db
        plt.title(f"Iteration {i+1}")
        plt.scatter(x,y)
        plt.plot(w*x + b,y)
        plt.show()

X = np.random.uniform(1, 3, 6)
Y = 1.7 + 0.5 * X
w_in = 0.5
b_in = 1.7

alpha = 100
num_iterations = 3

gradient_descent(X,Y,w_in,b_in,alpha,num_iterations,compute_gradient)