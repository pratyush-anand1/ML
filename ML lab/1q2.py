from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Boston.csv') #loading the dataset by giving its location
x = dataset.iloc[:,1:13].values
y = dataset.iloc[:,-1].values

#we split our data into train and test data in 80:20 ratio to test our model on test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

regressor = LinearRegression()
regressor.fit(x,y)

y_pred = regressor.predict(x_test)
print("Actual dataset values followd by the predicted price:")
print(y_test)
print(y_pred)