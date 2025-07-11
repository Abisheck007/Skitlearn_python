import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics,neighbors

df = pd.read_csv('skitlearn/Classification/laptop_buying.csv')

le = LabelEncoder()
x = df[['Age',
        'Income']].values
for i in range(len(x[0])):
    x[:,i] = le.fit_transform(x[:,i])
# print(x)

y  = df[['class']].replace({
    'Yes':1,
    'No':0
})
# print(y)
#model creation
knn = neighbors.KNeighborsClassifier(n_neighbors=3,weights="uniform")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test,prediction)
print("accuracy:", accuracy)
print("prediction:", prediction)