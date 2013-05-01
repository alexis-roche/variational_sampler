import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from ..variational_sampler import VariationalSampler
from ..gaussian import (Gaussian, FactorGaussian)


def target1d(x):
    return -.5 * x ** 2


def target(x):
    return -.5 * np.sum(x ** 2, 0)


def _test_basic(vs, objective='kl'):
    f = vs.fit(objective=objective)
    print(f.theta)
    print(f.fit)
    print(f.var_moment)
    print(f.fisher_info)
    print(f.var_theta)
    print(f.kl_error)


def test1d_vs_basic():
    _test_basic(VariationalSampler(target1d, (0, 1), 10))


def test2d_vs_basic():
    _test_basic(VariationalSampler(target, (np.zeros(2), np.eye(2)), 50))


def test1d_is_basic():
    _test_basic(VariationalSampler(target1d, (0, 1), 10),
                objective='l')


def test2d_is_basic():
    _test_basic(VariationalSampler(target, (np.zeros(2), np.eye(2)), 50),
                objective='l')


def test_loss():
    vs = VariationalSampler(target1d, (0, 1), 10)
    f = vs.fit()
    f._cache['qe'][:] = 0
    f._cache['log_qe'][:] = -np.inf
    f._cache['theta'] = None
    assert_equal(f._loss(None), np.inf)
    print(f._loss(np.array((0, 0, -2500))))
    print(f._loss(np.array((0, 0, -1e10))))


def _test_vs_exactness(dim):
    m = np.zeros(dim)
    A = np.random.random((dim, dim))
    V = np.dot(A, A.T)
    g = Gaussian(m, V)
    log_g = lambda x: g.log(x)
    ndraws = ((dim + 1) * (dim + 2)) / 2
    vs = VariationalSampler(log_g, g, ndraws)
    f = vs.fit()
    assert_almost_equal(f.fit.Z, 1, decimal=2)
    assert_array_almost_equal(f.fit.m, m, decimal=2)
    assert_array_almost_equal(f.fit.V, V, decimal=2)


def test_vs_exactness_2d():
    _test_vs_exactness(2)


def test_vs_exactness_3d():
    _test_vs_exactness(3)


def test_vs_exactness_5d():
    _test_vs_exactness(5)


def _test_vs_exactness_factor_gauss(dim):
    m = np.zeros(dim)
    v = np.random.random(dim) ** 2
    g = FactorGaussian(m, v)
    log_g = lambda x: g.log(x)
    ndraws = 2 * dim + 1
    vs = VariationalSampler(log_g, g, ndraws)
    f = vs.fit(family='factor_gaussian')
    assert_almost_equal(f.fit.Z, 1, decimal=2)
    assert_array_almost_equal(f.fit.m, m, decimal=2)
    assert_array_almost_equal(f.fit.v, v, decimal=2)


def test_vs_exactness_factor_gauss_2d():
    _test_vs_exactness_factor_gauss(2)


def test_vs_exactness_factor_gauss_3d():
    _test_vs_exactness_factor_gauss(3)


def test_vs_exactness_factor_gauss_5d():
    _test_vs_exactness_factor_gauss(5)


def test1d_vs_constant_kernel():
    _test_basic(VariationalSampler(target1d, (1, 2), 10))


def test2d_vs_constant_kernel():
    _test_basic(VariationalSampler(target, (np.ones(2), 2 * np.eye(2)), 50))


def test1d_is_constant_kernel():
    _test_basic(VariationalSampler(target1d, (1, 2), 10),
                objective='l')


def test2d_is_constant_kernel():
    _test_basic(VariationalSampler(target, (np.ones(2), 2 * np.eye(2)), 50),
                objective='l')


def test1d_vs_custom_kernel():
    _test_basic(VariationalSampler(target1d, (1, 2), 10))


def test2d_vs_custom_kernel():
    _test_basic(VariationalSampler(target, (np.ones(2), 2 * np.eye(2)), 50))


def test1d_is_custom_kernel():
    _test_basic(VariationalSampler(target1d, (1, 2), 10))
                

def test2d_is_custom_kernel():
    _test_basic(VariationalSampler(target, (np.ones(2), 2 * np.eye(2)), 50))
