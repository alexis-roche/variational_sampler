import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..sampling import Sample


def test1d_basic():
    s = Sample((0, 1), ndraws=10)
    assert_equal(s.x.size, 10)


def test1d_with_reflection():
    s = Sample((0, 1), ndraws=10, reflect=True)
    assert_equal(s.x.size, 20)
    assert_almost_equal(s.x.sum(), 0)


def test1d_custom_generator():
    s = Sample((0, 1), generator=(1, 2), ndraws=10, reflect=True)
    assert_equal(s.x.size, 20)
    assert_array_equal(0 * s.w, np.zeros(20))

def test2d_basic():
    s = Sample((np.zeros(2), np.eye(2)), ndraws=10)
    assert_equal(s.x.shape, (2, 10))
