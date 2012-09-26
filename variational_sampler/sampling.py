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
        raiseValueError('input not understood')
    return G


class Sample(object):

    def __init__(self, kernel, ndraws=None, reflect=False):
        """
        Instantiate Sample class.

        Given a kernel w(x), a sample (x1, x2, ..., xn) is generated
        such that, for any function f(x), the integral

        int w(x)f(x)dx 

        is approximated by the empirical mean

        (1/n) sum f(xi)

        Parameters
        ----------
        kernel: tuple
          a tuple (ms, Vs) where ms is a vector representing the mean
          of the sampling kernel and Vs is a matrix or vector
          representing the variance (if a vector, then a diagonal
          variance is assumed)
        """
        self.kernel = as_gaussian(kernel)
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
        self.w = None
