import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors , metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


cardata = pd.read_csv("skitlearn/Classification/car.data")

# print(cardata.head())

x = cardata[['buying',
             'maint',
             'safety']].values

y = cardata[['class']]
le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = le.fit_transform(x[:, i])
# print(x)

y = cardata[['class']].replace({
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 4
})
# print(y)
knn = neighbors.KNeighborsClassifier(n_neighbors=3,weights='uniform')
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)
print("Prediction:", prediction)
print(plt.plot(x,y))

