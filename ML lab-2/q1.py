import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.array([0, 1, 2, 3, 1, 1.5]).reshape(-1, 1)
y = np.array([1, 2, 1.5, 2, 1, 2])

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

coefficients = lin_reg.coef_
print(f"Value of a:{coefficients[2]},value of b:{coefficients[1]},value of c:{coefficients[0]}")
y_pred = lin_reg.predict(x_poly)

error = np.sum((y - y_pred) ** 2)

print(f"Sum of Squared Error: {error}")