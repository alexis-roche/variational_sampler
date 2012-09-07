from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal

from ..variational_sampler import VariationalSampler


def test1():
    VS = VariationalSampler()
    assert_equal(VS.x, 0)
