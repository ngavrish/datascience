import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../course/Part1/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.iloc[0, :])-1].values


# training set and test state
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""