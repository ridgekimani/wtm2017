import numpy as np

"""
Covariance is the measure in which two variables change together
"""


def co_variance():
    x = [6, 8, 10, 14, 18]
    y = [7, 9, 13, 17.5, 18]
    cov = np.cov(m=x, y=y)[0][1]
    return cov

if __name__ == '__main__':
    print('The covariance is ', co_variance())
