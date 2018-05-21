# Support Vector Machine (SVM)

# Importing the libraries
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('./datasets/train.csv')
# passanger id = 0, class = 1, sex  = 2, age = 3, siblings = 4, parch = 5, fare = 6, embarked = 7
test_dataset = pd.read_csv('./datasets/test.csv')

# taking care of missing data
from sklearn_pandas.categorical_imputer import CategoricalImputer
imputer = CategoricalImputer(missing_values='NaN')
imputer.fit(train_dataset["Sex"])
imputer.fit(test_dataset["Sex"])
imputer.fit(train_dataset["Embarked"])
imputer.fit(test_dataset["Embarked"])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='most_frequent')
imputer.fit(train_dataset["PassengerId"].reshape(-1,1))
imputer.fit(test_dataset)


X_test = test_dataset.iloc[:, [0, 1, 3, 4, 5, 6, 8, 10]].values
X_train = train_dataset.iloc[:, [0, 2, 4, 5, 6, 7, 9, 11]].values
y_train = train_dataset.iloc[:, 1].values

print(str(X_train[7]))
# Feature Scaling
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
sc = StandardScaler()
maxAbsScaler = MaxAbsScaler()
# age scaling
X_train[:, [3]] = sc.fit_transform(X_train[:, [3]])
X_test[:, [3]] = sc.transform(X_test[:, [3]])
# fare scaling
X_train[:, [6]] = maxAbsScaler.fit_transform(X_train[:, [6]])
X_test[:, [6]] = maxAbsScaler.transform(X_test[:, [6]])
# siblings scaling

# Categorical data fix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder= LabelEncoder()
X_train[:, 2] = labelencoder.fit_transform(X_train[:, 2])
X_test[:, 2] = labelencoder.fit_transform(X_test[:, 2])

print("Before label encoding = " + str(X_train[:, 7]))
X_train[:, 7] = labelencoder.fit_transform(X_train[:, 7])
X_test[:, 7] = labelencoder.fit_transform(X_test[:, 7])


df = pd.DataFrame(X_train)

onehotencoder = OneHotEncoder(categorical_features=[1, 7])

X_train = onehotencoder.fit_transform(X_train[:, :]).toarray()
X_test = onehotencoder.fit_transform(X_test[:, :]).toarray()


# Fitting SVM to the Training set
from sklearn.svm import SVC
#
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(str(y_pred))
submission = pd.DataFrame({
        "PassengerId": test_dataset["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv(f"./submission_1.csv", index=False)
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
