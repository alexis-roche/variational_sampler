import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..gaussian import Gaussian

TWO_PI = 2 * np.pi
SQRT_TWO_PI = np.sqrt(TWO_PI)


def test1d_basic():
    g = Gaussian(0, 1)
    assert_equal(g.m, 0)
    assert_equal(g.V, 1)
    assert_almost_equal(g.Z, 1)
    assert_almost_equal(g.K, 1 / SQRT_TWO_PI)


def test1d_basic2():
    g = Gaussian(0, 4)
    assert_equal(g.m, 0)
    assert_equal(g.V, 4)
    assert_almost_equal(g.Z, 1)
    assert_almost_equal(g.K, 1 / (2 * SQRT_TWO_PI))


def test1d_input_factor():
    g = Gaussian(0, 1, K=0.56)
    assert_equal(g.m, 0)
    assert_equal(g.V, 1)
    assert_almost_equal(g.K, 0.56)


def test1d_input_factor2():
    g = Gaussian(0, 1, Z=SQRT_TWO_PI)
    assert_equal(g.m, 0)
    assert_equal(g.V, 1)
    assert_almost_equal(g.K, 1)


def test1d_input_natural_params():
    g = Gaussian(theta=[-.5, 0, 0])
    assert_equal(g.m, 0)
    assert_equal(g.V, 1)
    assert_almost_equal(g.K, 1)


def test2d_basic():
    g = Gaussian([0, 0], np.eye(2))
    assert_array_equal(g.m, [0, 0])
    assert_array_equal(g.V, np.eye(2))
    assert_almost_equal(g.Z, 1)
    assert_almost_equal(g.K, 1 / TWO_PI)


def test2d_input_natural_params():
    g = Gaussian(theta=[-.5, 0, -.5, 0, 0, 0])
    assert_array_equal(g.m, [0, 0])
    assert_array_equal(g.V, np.eye(2))
    assert_almost_equal(g.Z, TWO_PI)
    assert_almost_equal(g.K, 1)

