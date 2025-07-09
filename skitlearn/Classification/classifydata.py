from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Hours of study vs good/bad grades for 10 different students


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
print("x_train:", X_train.shape)
print("x_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
 