"""
Quadratic regression, or regression  with a second order regression is given by

a = alpha + b1x + b2 x **2
 

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def predict():
    x_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]

    x_test = [[6], [8], [11], [16]]
    y_test = [[8], [12], [15], [18]]
    regr = LinearRegression()
    regr.fit(x_train, y_train)

    def quadratic_regressor():
        xx = np.linspace(0, 26, 100)
        yy = regr.predict(xx.reshape(xx.shape[0], 1))
        plt.plot(xx, yy)
        quad = PolynomialFeatures(degree=2)
        x_train_quad = quad.fit_transform(x_train)
        x_test_quad = quad.transform(x_test)
        regressor = LinearRegression()
        regressor.fit(x_train_quad, y_train)
        print('Simple', regr.score(x_test, y_test))
        print('Quadratic', regressor.score(x_test_quad, y_test))
        xx_quadratic = quad.transform(xx.reshape(xx. shape[0], 1))
        plt.plot(xx, regressor.predict(xx_quadratic), c='r', linestyle='--')
        plt.title('Pizza regressed')
        plt.xlabel('Diameter in inches')
        plt.ylabel('Price in dollars')
        plt.axis([0, 25, 0, 25])
        plt.grid(True)
        plt.scatter(x_train, y_train)
        plt.show()

    return quadratic_regressor()


if __name__ == '__main__':
    predict()
