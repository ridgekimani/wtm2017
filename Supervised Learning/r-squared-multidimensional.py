from sklearn.linear_model import LinearRegression


def predict():
    x = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
    y = [[7], [9], [13], [17.5], [18]]

    x_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
    y_test = [[11], [8.5], [15], [18], [11]]

    regr = LinearRegression()

    regr.fit(x, y)

    predictions = regr.predict(x_test)

    for i, prediction in enumerate(predictions):
        print("Predicted %s, Target %s" % (prediction, y_test[i]))

    return regr.score(x_test, y_test)


if __name__ == '__main__':
    print("The score is ", predict())
