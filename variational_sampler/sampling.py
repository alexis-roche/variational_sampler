"""
Independence sampling
"""
from time import time
import numpy as np
import warnings

from .numlib import force_tiny
from .gaussian import Gaussian, FactorGaussian


def reflect_sample(xs):
    return np.reshape(np.array([xs.T, -xs.T]).T,
                      [xs.shape[0], 2 * xs.shape[1]])


def as_gaussian(g):
    if isinstance(g, Gaussian) or isinstance(g, FactorGaussian):
        return g
    try:
        m = np.asarray(g[0])
        V = np.asarray(g[1])
        if V.ndim < 2:
            G = FactorGaussian(m, V)
        elif V.ndim == 2:
            G = Gaussian(m, V)
        else:
            raise ValueError('input variance not understood')
    except:
        raise ValueError('input not understood')
    return G


def sample_fun(f, x):
    try:
        ff = f
        y = f(x).squeeze()
    except:
        ff = lambda x: np.array([f(xi) for xi in x.T])
        y = ff(x).squeeze()
    return y, ff


class Sample(object):

    def __init__(self, target, kernel, context=None, ndraws=None, reflect=False):
        """
        Instantiate Sample class.

        Given a context w(x), a sample (x1, x2, ..., xn) is generated
        such that, for any function f(x), the integral

        int w(x)f(x)dx 

        is approximated by the empirical mean

        (1/n) sum ci f(xi),

        where ci are appropriate weights.

        Parameters
        ----------
        target: callable
          returns the log of the target distribution

        kernel: tuple
          a tuple (ms, Vs) where ms is a vector representing the mean
          of the sampling distribution and Vs is a matrix or vector
          representing the variance (if a vector, then a diagonal
          variance is assumed)
        """
        self.kernel = as_gaussian(kernel)
        self.target = target
        if context is None:
            self.context = None
        elif context == 'kernel':
            self.context = self.kernel
        else:
            self.context = as_gaussian(context)
        self.reflect = reflect
        if ndraws == None:
            ndraws = self.kernel.theta_dim
            warnings.warn('Setting ndraws to the minimum number: %d' % ndraws)

        # Sample random points
        t0 = time()
        self._sample(ndraws)
        self.time = time() - t0

    def _sample(self, ndraws):
        """
        Sample independent points from the specified kernel and
        compute associated distribution values.
        """
        self.x = self.kernel.sample(ndraws=ndraws)
        if self.reflect:
            self.x = reflect_sample(self.x)
        self.log_p, self.target = sample_fun(self.target, self.x)
        if self.context is self.kernel:
            self.log_w = np.zeros(self.log_p.size)
        elif self.context is None:
            self.log_w = -self.kernel.log(self.x)
        else:
            self.log_w = self.context.log(self.x) - self.kernel.log(self.x)

