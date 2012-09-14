import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from ..variational_sampler import (VariationalSampler,
                                   VariationalSamplerIS)
from ..gaussian import (Gaussian, FactorGaussian)


def target1d(x):
    return np.exp(-.5 * x ** 2)


def target(x):
    return np.exp(-.5 * np.sum(x ** 2, 0))


def _test_basic(vs):
    print vs.theta
    print vs.fit
    print vs.loc_fit
    print vs.sigma1
    print vs.sigma2
    print vs.kl_error


def test1d_vs_basic():
    _test_basic(VariationalSampler(target1d, 0, 1, ndraws=10))


def test2d_vs_basic():
    _test_basic(VariationalSampler(target, np.zeros(2), np.eye(2), ndraws=50))


def test1d_ds_basic():
    _test_basic(VariationalSamplerIS(target1d, 0, 1, ndraws=10))


def test2d_ds_basic():
    _test_basic(VariationalSamplerIS(target, np.zeros(2), np.eye(2), ndraws=50))


def test_loss():
    vs = VariationalSampler(target1d, 0, 1, ndraws=10)
    vs._cache['q'][:] = 0
    vs._cache['log_q'][:] = -np.inf
    vs._cache['theta'] = None
    assert_equal(vs._loss(None), np.inf)
    print vs._loss(np.array((0, 0, -2500)))
    print vs._loss(np.array((0, 0, -1e10)))


def _test_vs_exactness(dim):
    m = np.zeros(dim)
    A = np.random.random((dim, dim))
    V = np.dot(A, A.T)
    g = Gaussian(m, V)
    ndraws = ((dim + 1) * (dim + 2)) / 2
    vs = VariationalSampler(g, m, V, ndraws=ndraws)
    assert_almost_equal(vs.fit.Z, 1, decimal=2)
    assert_array_almost_equal(vs.fit.m, m, decimal=2)
    assert_array_almost_equal(vs.fit.V, V, decimal=2)


def test_vs_exactness_2d():
    _test_vs_exactness(2)


def test_vs_exactness_3d():
    _test_vs_exactness(3)


def test_vs_exactness_5d():
    _test_vs_exactness(5)


def _test_vs_exactness_fg(dim):
    m = np.zeros(dim)
    v = np.random.random(dim) ** 2
    g = FactorGaussian(m, v)
    ndraws = 2 * dim + 1
    vs = VariationalSampler(g, m, v, ndraws=ndraws,
                            family='factor_gaussian')
    assert_almost_equal(vs.fit.Z, 1, decimal=2)
    assert_array_almost_equal(vs.fit.m, m, decimal=2)
    assert_array_almost_equal(vs.fit.v, v, decimal=2)


def test_vs_exactness_fg_2d():
    _test_vs_exactness_fg(2)


def test_vs_exactness_fg_3d():
    _test_vs_exactness_fg(3)


def test_vs_exactness_fg_5d():
    _test_vs_exactness_fg(5)

