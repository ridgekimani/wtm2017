from sklearn import linear_model


def example(value):
    x = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    return regr.predict(X=value)


if __name__ == '__main__':
    print(example(12))
