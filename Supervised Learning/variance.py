import numpy as np


"""
Variance is the measure in which data is spread altogether 
"""


def calculate_variance():
    x = [6, 8, 10, 14, 18]
    variance = np.var(x, ddof=1)  # Delta degrees of freedom
    return variance

if __name__ == "__main__":
    print("Variance is ", calculate_variance())
