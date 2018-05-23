# Support Vector Machine (SVM)

# Importing the libraries
import numpy
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('./datasets/train.csv')
test_dataset = pd.read_csv('./datasets/test.csv')
# class = 0, sex  = 1, age = 2, siblings = 3, parch = 4, fare = 5, embarked = 6
X_test = test_dataset.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values
X_train = train_dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
y_train = train_dataset.iloc[:, 1].values

# taking care of missing data
from sklearn_pandas.categorical_imputer import CategoricalImputer
categorical_imputer = CategoricalImputer(missing_values='NaN')

categorical_imputer.fit(X_train[:, 1])
X_train[:, 1] = categorical_imputer.transform(X_train[:, 1])

categorical_imputer.fit(X_test[:, 1])
X_test[:, 1] = categorical_imputer.transform(X_test[:, 1])

categorical_imputer.fit(X_train[:, 6])
X_train[:, 6] = categorical_imputer.transform(X_train[:, 6])

categorical_imputer.fit(X_test[:, 6])
X_test[:, 6] = categorical_imputer.transform(X_test[:, 6])


from sklearn.preprocessing import Imputer
train_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_imputer.fit(X_train[:, 2:6])
X_train[:, 2:6] = train_imputer.transform(X_train[:, 2:6])


test_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
test_imputer.fit(X_test[:, 2:6])
X_test[:, 2:6] = test_imputer.transform(X_test[:, 2:6])

# Feature Scaling
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
sc = StandardScaler()
maxAbsScaler = MaxAbsScaler()
# age scaling
X_train[:, [2]] = sc.fit_transform(X_train[:, [2]])
X_test[:, [2]] = sc.transform(X_test[:, [2]])
# fare scaling
X_train[:, [5]] = maxAbsScaler.fit_transform(X_train[:, [5]])
X_test[:, [5]] = maxAbsScaler.transform(X_test[:, [5]])
# siblings scaling

# Categorical data fix
df_before_categorical = pd.DataFrame(X_train)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder= LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])
X_test[:, 1] = labelencoder.fit_transform(X_test[:, 1])

X_train[:, 6] = labelencoder.fit_transform(X_train[:, 6])
X_test[:, 6] = labelencoder.fit_transform(X_test[:, 6])


df = pd.DataFrame(X_train)

onehotencoder = OneHotEncoder(categorical_features=[0, 6])

X_train = onehotencoder.fit_transform(X_train[:, :]).toarray()
X_test = onehotencoder.fit_transform(X_test[:, :]).toarray()

# Fitting SVM to the Training set
from sklearn.svm import SVC
#
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(str(y_pred))
submission = pd.DataFrame({
        "PassengerId": test_dataset["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv(f"./submission.csv", index=False)
# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
#
# cm = confusion_matrix(y_test, y_pred)
#
# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
#
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('SVM (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
#
# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
#
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('SVM (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
