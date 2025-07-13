from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd

df = pd.read_csv('skitlearn/Linear_regression/student_scores.csv')

x = df[['Hours_Studied']]
y = df[['Marks_Scored']]

x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.2)

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_test)
performance = model.score(x,y)
print("prediction",prediction)
print("performance",performance)
print("intercept",model.intercept_)
print("coef",model.coef_)
print
