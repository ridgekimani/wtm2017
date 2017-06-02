from sklearn.linear_model import LinearRegression


def r_squared():
    x = [[6], [8], [10], [14], [18]]
    y = [[7], [9], [13], [17.5], [18]]
    x_test = [[8], [9], [11], [16], [12]]
    y_test = [[11], [8.5], [15], [18], [11]]

    def score():
        regr = LinearRegression()
        regr.fit(x, y)
        return regr.score(x_test, y_test)

    return score()


if __name__ == '__main__':
    print("The score of the linear regression is ", r_squared())
