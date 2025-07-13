import pandas as pd 
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
import matplotlib.pyplot as plt

data = load_breast_cancer()
x = data.data
y = data.target

xtrain,xtest,ytrain,ytest  = train_test_split(x,y,test_size=0.2)
model = KMeans(n_clusters=3,random_state=0)
model.fit(xtrain)##it is common in clustering we only pass one train set to fit in
predictions = model.predict(xtest)
label = model.labels_
accuracy = metrics.accuracy_score(ytest,predictions)
print("actual:",ytest)
print("pred:",predictions)
print('label:',label)
print('acc:',accuracy)
plt.plot(label)
plt.show()
print(pd.crosstab(ytrain,label))