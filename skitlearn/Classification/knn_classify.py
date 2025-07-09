import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors , metrics
from sklearn.preprocessing import LabelEncoder

cardata = pd.read_csv("car.data")

print(cardata.head())