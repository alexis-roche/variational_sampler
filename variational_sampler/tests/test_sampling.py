import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..sampling import Sample


def target1d(x):
    return np.exp(-.5 * x ** 2)


def target(x):
    return np.exp(-.5 * np.sum(x ** 2, 0))


def test1d_basic():
    s = Sample(target1d, 0, 1, ndraws=10)
    assert_equal(s.x.size, 10)
    assert_equal(s.p.size, 10)


def test1d_with_reflection():
    s = Sample(target1d, 0, 1, ndraws=10, reflect=True)
    assert_equal(s.x.size, 20)
    assert_equal(s.p.size, 20)
    assert_almost_equal(s.x.sum(), 0)


def test2d_basic():
    s = Sample(target, np.zeros(2), np.eye(2), ndraws=10)
    assert_equal(s.x.shape, (2, 10))
    assert_equal(s.p.size, 10)
