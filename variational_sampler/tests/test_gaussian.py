import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..gaussian import (Gaussian,
                        FactorGaussian,
                        GaussianFamily,
                        FactorGaussianFamily)

TWO_PI = 2 * np.pi
SQRT_TWO_PI = np.sqrt(TWO_PI)


def test1d_basic():
    for c in (Gaussian, FactorGaussian):
        g = c(0, 1)
        assert_equal(g.m, 0)
        assert_equal(g.V, 1)
        assert_almost_equal(g.Z, 1)
        assert_almost_equal(g.K, 1 / SQRT_TWO_PI)


def test1d_basic2():
    for c in (Gaussian, FactorGaussian):
        g = c(0, 4)
        assert_equal(g.m, 0)
        assert_equal(g.V, 4)
        assert_almost_equal(g.Z, 1)
        assert_almost_equal(g.K, 1 / (2 * SQRT_TWO_PI))


def test1d_input_factor():
    for c in (Gaussian, FactorGaussian):
        g = c(0, 1, K=0.56)
        assert_equal(g.m, 0)
        assert_equal(g.V, 1)
        assert_almost_equal(g.K, 0.56)
        

def test1d_input_factor2():
    for c in (Gaussian, FactorGaussian):
        g = c(0, 1, Z=SQRT_TWO_PI)
        assert_equal(g.m, 0)
        assert_equal(g.V, 1)
        assert_almost_equal(g.K, 1)


def test1d_input_natural_params():
    for c in (Gaussian, FactorGaussian):
        g = c(theta=[-.5, 0, 0])
        assert_equal(g.m, 0)
        assert_equal(g.V, 1)
        assert_almost_equal(g.K, 1)


def test2d_basic():
    for c, v in zip((Gaussian, FactorGaussian),
                    (np.eye(2), np.ones(2))):
        g = c([0, 0], v)
        assert_array_equal(g.m, [0, 0])
        assert_array_equal(g.V, np.eye(2))
        assert_almost_equal(g.Z, 1)
        assert_almost_equal(g.K, 1 / TWO_PI)


def test2d_input_natural_params():
    for c, theta in zip((Gaussian, FactorGaussian),
                        ([-.5, 0, -.5, 0, 0, 0], [-.5, -.5, 0, 0, 0])):
        g = c(theta=theta)
        assert_array_equal(g.m, [0, 0])
        assert_array_equal(g.V, np.eye(2))
        assert_almost_equal(g.Z, TWO_PI)
        assert_almost_equal(g.K, 1)


def test_rescale():
    for c, v in zip((Gaussian, FactorGaussian),
                    (np.eye(2), np.ones(2))):
        g = c([0, 0], v)
        K = g.K
        g.rescale(10)
        assert_array_equal(g.m, [0, 0])
        assert_array_equal(g.V, np.eye(2))
        assert_array_equal(g.K, 10 * K)


def test_multiply():
    for c, v in zip((Gaussian, FactorGaussian),
                    (np.eye(2), np.ones(2))):
        g1 = c([0, 0], v)
        g2 = c([0, 0], v)
        g = g1 * g2
        assert_array_equal(g.m, [0, 0])
        assert_array_equal(g.V, .5 * np.eye(2))


def test_multiply_hybrid():
    g1 = Gaussian([0, 0], np.eye(2))
    g2 = FactorGaussian([0, 0], np.ones(2))
    g = g1 * g2
    assert_array_equal(g.m, [0, 0])
    assert_array_equal(g.V, .5 * np.eye(2))


def test_multiply_hybrid_reverse():
    g1 = FactorGaussian([0, 0], np.ones(2))
    g2 = Gaussian([0, 0], np.eye(2))
    g = g1 * g2
    assert_array_equal(g.m, [0, 0])
    assert_array_equal(g.V, .5 * np.eye(2))


def test_power():
    for c, v in zip((Gaussian, FactorGaussian),
                    (np.eye(2), np.ones(2))):
        g = c([0, 0], v) ** 2
        assert_array_equal(g.m, [0, 0])
        assert_array_equal(g.V, .5 * np.eye(2))


def test_gaussian_family():
    f = GaussianFamily(5)
    m = np.random.rand(5)
    m = np.zeros(5)

    V = np.random.rand(5, 5)
    V = np.dot(V.T, V)
    E = V + np.dot(m.reshape((5, 1)), m.reshape((1, 5)))
    moment = np.concatenate((E[np.triu_indices(5)], m, (1,)))
    g = f.from_moment(moment)

    print ('***********************')
    print g.Z
    print g.m
    print g.V

    assert_almost_equal(g.Z, 1)
    assert_array_almost_equal(g.m, m)
    assert_array_almost_equal(g.V, V)


def test_factor_gaussian_family():
    f = FactorGaussianFamily(5)
    Z = 1
    m = np.random.rand(5)
    v = np.random.rand(5) ** 2
    e = v + m ** 2
    moment = np.concatenate((e, m, (1,)))
    g = f.from_moment(moment)
    assert_almost_equal(g.Z, 1)
    assert_array_almost_equal(g.m, m)
    assert_array_almost_equal(g.v, v)

