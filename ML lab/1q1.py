import numpy as np
import matplotlib.pyplot as plt

#this function computes partial derivatives used in the gradient descent formula
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

    return w,b

x_train = np.array([0,1,2,3,4,5,6,7,8,9])
y_train = np.array([1,3,2,5,7,8,8,9,10,12])
aplha = 0.01 #initialising learning rate aplha
w_in = 0
b_in = 0
iterations = 1500

w_fin,b_fin = gradient_descent(x_train,y_train,w_in,b_in,aplha,iterations,compute_gradient)
print(f"The best fit line for given dataset is f(x) = {w_fin :8.4f}x + {b_fin:8.4f}")

plt.scatter(x_train,y_train,label = 'Data')
plt.plot(x_train,w_fin*x_train + b_fin,label = 'Best fit Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plotting the best-fit line')
plt.show()