"""
Variational sampling
"""
from time import time
import numpy as np

from .numlib import safe_exp
from .gaussian import Gaussian, FactorGaussian
from .kl_fit import KLFit
from .kl2_fit import KL2Fit
from .l_fit import LFit
from .gp_fit import GPFit


def reflect_sample(xs, m):
    return np.reshape(np.array([xs.T, m - xs.T]).T,
                      (xs.shape[0], 2 * xs.shape[1]))


def as_gaussian(g):
    """
    renormalize input to unit integral
    """
    if isinstance(g, Gaussian):
        return g.Z, Gaussian(g.m, g.V)
    elif isinstance(g, FactorGaussian):
        return g.Z, FactorGaussian(g.m, g.v)
    if len(g) == 2:
        Z = None
        m, V = np.asarray(g[0]), np.asarray(g[1])
    elif len(g) == 3:
        Z, m, V = float(g[0]), np.asarray(g[1]), np.asarray(g[2])
    else:
        raise ValueError('input not understood')
    if V.ndim < 2:
        G = FactorGaussian(m, V)
    elif V.ndim == 2:
        G = Gaussian(m, V)
    else:
        raise ValueError('input variance not understood')
    return Z, G


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
                 x=None, w=None):
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
        """
        self.Z, self.kernel = as_gaussian(kernel)
        self.target = target
        self.ndraws = ndraws
        self.reflect = reflect

        # Sample random points
        self.x, self.w = x, w
        t0 = time()
        self._sample()
        self.sampling_time = time() - t0

    def _sample(self, x=None, w=None):
        """
        Sample independent points from the specified kernel and
        compute associated distribution values.
        """
        if self.x == None:
            self.x = self.kernel.sample(ndraws=self.ndraws)
        else:
            self.x = np.reshape(self.x, (self.kernel.dim, len(self.x)))
        if self.reflect:
            self.x = reflect_sample(self.x, self.kernel.m)
        log_p, self.target = sample_fun(self.target, self.x)

        self.log_pe = log_p - self.kernel.log(self.x)
        self.pe, self.logscale = safe_exp(self.log_pe)
        self.log_pe -= self.logscale

        # This is a temporary HACK to implement a deterministic
        # version of VS.
        # The input weights are assumed to come from a
        # quadrature rule, so they need be multiplied by the number of
        # points for consistency with the random case where the
        # weights are conventionally one
        if not self.w == None:
            self.log_pe += np.log(self.w) + np.log(len(self.w))

    def fit(self, objective='kl', **args):
        """
        Perform fitting.

        Parameters
        ----------
        objective: str
          one of 'kl', 'l' or 'gp' standing for Kullback-Leibler
          divergence minimization, Likelihood maximization, and
          Gaussian Process fitting methods, respectively
        """
        if objective == 'kl':
            return KLFit(self, **args)
        elif objective == 'kl2':
            return KL2Fit(self, **args)
        elif objective == 'l':
            return LFit(self, **args)
        elif objective == 'gp':
            return GPFit(self, **args)
        else:
            raise ValueError('unknown objective')
