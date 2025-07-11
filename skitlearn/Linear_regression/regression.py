import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load dataset
data = fetch_california_housing()
x = data.data
y = data.target

model = LinearRegression()


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
prediction = model.predict(x_test)
performance_rate  = model.score(x,y)
print("prediction value is ",prediction)
print("performance rate ",performance_rate) 