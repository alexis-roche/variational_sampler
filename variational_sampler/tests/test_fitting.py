import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from ..variational_sampler import VariationalSampler
from ..importance_sampler import ImportanceSampler
from ..gaussian import (Gaussian, FactorGaussian)


def target1d(x):
    return -.5 * x ** 2


def target(x):
    return -.5 * np.sum(x ** 2, 0)


def _test_basic(vs):
    print(vs.theta)
    print(vs.fit)
    print(vs.loc_fit)
    print(vs.var_moment)
    print(vs.fisher_info)
    print(vs.var_theta)
    print(vs.kl_error)


def test1d_vs_basic():
    _test_basic(VariationalSampler(target1d, (0, 1), ndraws=10))


def test2d_vs_basic():
    _test_basic(VariationalSampler(target, (np.zeros(2), np.eye(2)), context='kernel', ndraws=50))


def test1d_cs_basic():
    _test_basic(ImportanceSampler(target1d, (0, 1), context='kernel', ndraws=10))


def test2d_cs_basic():
    _test_basic(ImportanceSampler(target, (np.zeros(2), np.eye(2)), context='kernel', ndraws=50))


def test_loss():
    vs = VariationalSampler(target1d, (0, 1), context='kernel', ndraws=10)
    vs._cache['qw'][:] = 0
    vs._cache['log_q'][:] = -np.inf
    vs._cache['theta'] = None
    assert_equal(vs._loss(None), np.inf)
    print(vs._loss(np.array((0, 0, -2500))))
    print(vs._loss(np.array((0, 0, -1e10))))


def _test_vs_exactness(dim):
    m = np.zeros(dim)
    A = np.random.random((dim, dim))
    V = np.dot(A, A.T)
    g = Gaussian(m, V)
    log_g = lambda x: g.log(x)
    ndraws = ((dim + 1) * (dim + 2)) / 2
    vs = VariationalSampler(log_g, g, ndraws=ndraws)
    assert_almost_equal(vs.fit.Z, 1, decimal=2)
    assert_array_almost_equal(vs.fit.m, m, decimal=2)
    assert_array_almost_equal(vs.fit.V, V, decimal=2)


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
    vs = VariationalSampler(log_g, g, ndraws=ndraws,
                            family='factor_gaussian')
    assert_almost_equal(vs.fit.Z, 1, decimal=2)
    assert_array_almost_equal(vs.fit.m, m, decimal=2)
    assert_array_almost_equal(vs.fit.v, v, decimal=2)


def test_vs_exactness_factor_gauss_2d():
    _test_vs_exactness_factor_gauss(2)


def test_vs_exactness_factor_gauss_3d():
    _test_vs_exactness_factor_gauss(3)


def test_vs_exactness_factor_gauss_5d():
    _test_vs_exactness_factor_gauss(5)


def test1d_vs_constant_kernel():
    _test_basic(VariationalSampler(target1d, (1, 2), 
                                   context=None,
                                   ndraws=10))

def test2d_vs_constant_kernel():
    _test_basic(VariationalSampler(target, (np.ones(2), 2 * np.eye(2)),
                                   context=None,
                                   ndraws=50))


def test1d_cs_constant_kernel():
    _test_basic(ImportanceSampler(target1d, (1, 2),
                                  context=None,
                                  ndraws=10))


def test2d_cs_constant_kernel():
    _test_basic(ImportanceSampler(target, (np.ones(2), 2 * np.eye(2)),
                                  context=None,
                                  ndraws=50))

def test1d_vs_custom_kernel():
    _test_basic(VariationalSampler(target1d, (1, 2), 
                                   context=(0, 1),
                                   ndraws=10))

def test2d_vs_custom_kernel():
    _test_basic(VariationalSampler(target, (np.ones(2), 2 * np.eye(2)),
                                   context=(np.zeros(2), np.eye(2)),
                                   ndraws=50))


def test1d_cs_custom_kernel():
    _test_basic(ImportanceSampler(target1d, (1, 2),
                                  context=(0, 1),
                                  ndraws=10))


def test2d_cs_custom_kernel():
    _test_basic(ImportanceSampler(target, (np.ones(2), 2 * np.eye(2)),
                                  context=(np.zeros(2), np.eye(2)),
                                  ndraws=50))
