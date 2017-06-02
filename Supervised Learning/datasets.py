import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
unique = np.unique(iris_y)

print(unique)

np.random.seed(0)
# Make a random permutation of the dataset that is present in the iris_X dataset
indices = np.random.permutation(len(iris_X))
X_train = iris_X[indices[:-10]]
y_train = iris_y[indices[:-10]]
X_test = iris_X[indices[-10:]]
y_test = iris_y[indices[-10:]]

"""Using the K-Neighbours Nearest example"""
knn = KNeighborsClassifier()
print(knn.fit(X_train, y_train))
print(knn.predict(X_test))
print(y_test)

"""The curse of dimensionality with the Diabetes dataset"""

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

"""The task in hand is to predict disease progression from physiological variables"""
regr = linear_model.LinearRegression()

print(regr.fit(diabetes_X_train, diabetes_y_train))

print(regr.coef_)

# Calculate the mean square error
print(np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

print(regr.score(diabetes_X_test, diabetes_y_test))

""" Shrinkage"""

X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0, 2].T

regr = linear_model.LinearRegression()

plt.figure()
np.random.seed(0)

for _ in range(6):
    this_X = .1 * np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s=3)

"""Ridge regression
    We can choose alpha to minimize the left out error
    The bias introduced by Ridge Regression is called Regularization    
    
"""
regr = linear_model.Ridge()
alphas = np.logspace(-4, -1, 6)
print([regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test)
       for alpha in alphas
       ])


"""Sparsity fitting only features 1 and 2"""
# indices = (0, 1)
#
# diabetes = datasets.load_diabetes()
#
# X_test = diabetes.data[:-20, indices]
# y_test = diabetes.target[:-20]
# X_train = diabetes.data[-20:, indices]
# y_train = diabetes.target[-20:]
#
# ols = linear_model.LinearRegression()
# ols.fit(X_train, y_train)
#
#
# def plot_figs(fig_num, elev, azim, X_train, clf):
#     fig = plt.figure(fig_num, figsize=(4, 3))
#     plt.clf()
#     ax = Axes3D(fig, elev=elev, azim=azim)
#
#     ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
#     ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
#                     np.array([[-.1, .15], [-.1, .15]]),
#                     clf.predict(np.array([[-.1, -.1, .15, .15],
#                                           [-.1, .15, -.1, .15]]).T
#                                 ).reshape((2, 2)),
#                     alpha=.5)
#     ax.set_xlabel('X_1')
#     ax.set_ylabel('X_2')
#     ax.set_zlabel('Y')
#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#
# # Generate the three different figures from different views
#
# elev = 43.5
# azim = -110
# plot_figs(1, elev, azim, X_train, ols)
#
# elev = -.5
# azim = 0
# plot_figs(2, elev, azim, X_train, ols)
#
# elev = -.5
# azim = 90
# plot_figs(3, elev, azim, X_train, ols)
#
# plt.show()




