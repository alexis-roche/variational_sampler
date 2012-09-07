import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal

from ..variational_sampler import VariationalSampler


def target1d(x):
    return np.exp(-.5 * x ** 2)


def target(x):
    return np.exp(-.5 * np.sum(x ** 2, 0))


def test1d_basic():
    vs = VariationalSampler(target1d, 0, 1, ndraws=10)
    g = vs.fit()
    print (g.K, g.m, g.V)


def test2d_basic():
    vs = VariationalSampler(target, np.zeros(2), np.eye(2), ndraws=50)
    g = vs.fit()
    print (g.K, g.m, g.V)

