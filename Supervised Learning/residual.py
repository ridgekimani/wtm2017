from sklearn import linear_model
import numpy as np


def residual_sum_of_squares():
    x = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]

    def example(value):
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        return regr.predict(value)

    print('The residual sum of squares is ', np.mean((example(x) - y) ** 2))


if __name__ == '__main__':
    residual_sum_of_squares()
