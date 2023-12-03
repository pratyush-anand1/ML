import pandas as pd
import numpy as np
import random
from sklearn.svm import SVC

def split_dataset(dataset, split_ratio):
    
    data_array = dataset.values
    np.random.shuffle(data_array)

    split_index = int(len(data_array) * split_ratio)
    training_set = data_array[:split_index]
    testing_set = data_array[split_index:]
    
    training_set = pd.DataFrame(training_set, columns=dataset.columns)
    testing_set = pd.DataFrame(testing_set, columns=dataset.columns)
    
    return training_set, testing_set

dataset = pd.read_csv('iris.csv')
training_set, test_set = split_dataset(dataset, 0.8)

x_train = training_set.iloc[:, :-1]
y_train = training_set.iloc[:, -1]

x_test = test_set.iloc[:, :-1]
y_test = test_set.iloc[:, -1]

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('Actual vs Predicted values:')
for i in range(len(y_test)):
    print(y_test[i]," " , y_pred[i])