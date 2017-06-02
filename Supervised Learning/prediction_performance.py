from sklearn import datasets, neighbors, linear_model


digits = datasets.load_digits()

X_digits = digits.data

y_digits = digits.target

# Using the K-NN model

knn = neighbors.KNeighborsClassifier()

knn.fit(X_digits, y_digits)

print(knn.score(X_digits, y_digits))

reg = linear_model.Ridge(alpha=.1)

reg.fit(X_digits, y_digits)

print(reg.score(X_digits, y_digits))