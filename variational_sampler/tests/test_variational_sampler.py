import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from numpy.testing import assert_array_almost_equal

from ..variational_sampler import (VariationalSampler,
                                   DirectSampler)


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
    _test_basic(DirectSampler(target1d, 0, 1, ndraws=10))


def test2d_ds_basic():
    _test_basic(DirectSampler(target, np.zeros(2), np.eye(2), ndraws=50))


def test_loss():
    vs = VariationalSampler(target1d, 0, 1, ndraws=10)
    vs._cache['q'][:] = 0
    vs._cache['log_q'][:] = -np.inf
    vs._cache['theta'] = None
    assert_equal(vs._loss(None), np.inf)
    print vs._loss(np.array((0, 0, -2500)))
    print vs._loss(np.array((0, 0, -1e10)))

