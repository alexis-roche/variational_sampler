"""
Variational sampling
"""
from time import time
import numpy as np
import warnings

from .numlib import force_tiny
from .gaussian import Gaussian, FactorGaussian
from .kl_fit import KLFit
from .naive_kl_fit import NaiveKLFit
from .gp_fit import GPFit


def reflect_sample(xs, m):
    return np.reshape(np.array([xs.T, m - xs.T]).T,
                      (xs.shape[0], 2 * xs.shape[1]))


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


class VariationalSampler(object):

    def __init__(self, target, kernel, ndraws, reflect=False,
                 context=None):
        """
        Variational sampler class.

        Fit a target distribution with a Gaussian distribution by
        maximizing an approximate KL divergence based on independent
        random sampling.

        Parameters
        ----------
        target: callable
          returns the log of the target distribution

        kernel: tuple
          a tuple `(m, V)` where `m` is a vector representing the mean
          of the sampling distribution and `V` is a matrix or vector
          representing the variance. If a vector, a diagonal variance
          is assumed.

        ndraws: int
          sample size

        reflect: bool
          if True, reflect the sample about the sampling kernel mean

        context: tuple
          a tuple `(m, V)` similar to the kernel argument that defines
          the local KL divergence used as a fitting objective. If
          None, the global KL divergence is used.
        """
        self.kernel = as_gaussian(kernel)
        self.target = target
        if context is None:
            self.context = None
        elif context == 'kernel':
            self.context = self.kernel
        else:
            self.context = as_gaussian(context)
        self.ndraws = ndraws
        self.reflect = reflect

        # Sample random points
        t0 = time()
        self._sample()
        self.sampling_time = time() - t0

    def _sample(self):
        """
        Sample independent points from the specified kernel and
        compute associated distribution values.
        """
        self.x = self.kernel.sample(ndraws=self.ndraws)
        if self.reflect:
            self.x = reflect_sample(self.x, self.kernel.m)
        self.log_p, self.target = sample_fun(self.target, self.x)
        if self.context is self.kernel:
            self.log_w = np.zeros(self.log_p.size)
        elif self.context is None:
            self.log_w = -self.kernel.log(self.x)
        else:
            self.log_w = self.context.log(self.x) - self.kernel.log(self.x)

    def fit(self, objective='kl', **args):
        """
        Perform fitting.
        
        Parameters
        ----------
        objective: str
          one of 'kl', 'naive_kl' or 'gp'
        """
        if objective == 'kl':
            return KLFit(self, **args)
        elif objective == 'naive_kl':
            return NaiveKLFit(self, **args)
        elif objective == 'gp':
            return GPFit(self, **args)
        else:
            raise ValueError('unknown objective')
        
