import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..variational_sampler import VariationalSampler


def target1d(x):
    return -.5 * x ** 2


def target(x):
    return -.5 * np.sum(x ** 2, 0)


def test1d_basic():
    s = VariationalSampler(target1d, (0, 1), 10)
    assert_equal(s.x.size, 10)


def test1d_with_reflection():
    s = VariationalSampler(target1d, (0, 1), 10, reflect=True, context='kernel')
    assert_equal(s.x.size, 20)
    assert_almost_equal(s.x.sum(), 0)


def test1d_custom_generator():
    s = VariationalSampler(target1d, (1, 2), 10, reflect=True, context=(0, 1))
    assert_equal(s.x.size, 20)


def test2d_basic():
    s = VariationalSampler(target, (np.zeros(2), np.eye(2)), 10, context='kernel')
    assert_equal(s.x.shape, (2, 10))
