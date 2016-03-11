import pandas
import numpy
import matplotlib.pyplot as plt


def univariate_stats():
    num_examples = 1000
    z = pandas.Series(numpy.random.randn(num_examples))

    # Minimum
    print(z.min())
    # Maximum
    print(z.max())
    # Mean
    print(z.mean())
    # Median
    print(z.median())
    # Variance
    print(z.var())
    # Standard deviation
    print(z.std())
    # Mean absolute deviation
    print(z.mad())
    # Interquartile range
    print(z.quantile(0.75) - z.quantile(0.25))

    z.plot(kind="hist")


def multivariate_stats():
    num_examples = 1000
    x = pandas.Series(numpy.random.randn(num_examples))
    y = x + pandas.Series(numpy.random.randn(num_examples))
    z = x + pandas.Series(numpy.random.randn(num_examples))

    # Covariance
    print(y.cov(z))

    # Covariance of y with itself is equal to variance
    print(y.cov(y), y.var())

    # Correlation
    print(y.corr(z))

univariate_stats()
multivariate_stats()

plt.show()
