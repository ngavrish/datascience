# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/802619/Projects/datascience/superdatascience/src/multiple_linear_regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid dummy variable trap

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train)

# fitting multiple linear regression to trainig set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(np.ones((50, 1)).astype(int), X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_ols.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_ols.summary())

X_opt = X[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_ols.summary())

X_opt = X[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_ols.summary())


X_opt = X[:, [0, 3]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_ols.summary())
