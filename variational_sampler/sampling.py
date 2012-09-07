"""
Independence sampling, annealed independence sampling.
"""
from time import time
import numpy as np
import warnings

from .numlib import force_tiny
from .gaussian import Gaussian


def reflect_sample(xs):
    return np.reshape(np.array([xs.T, -xs.T]).T,
                      [xs.shape[0], 2 * xs.shape[1]])


class Sample(object):

    def __init__(self, target, ms, Vs, ndraws=None, reflect=False):
        """
        Instantiate Sample class.

        Parameters
        ----------
        target : function
          target distribution
        ms : vector
          mean of the sampling kernel
        Vs : matrix
          variance of the sampling kernel

        """
        self._initialize(target, ms, Vs, ndraws, reflect)

    def _initialize(self, target, ms, Vs, ndraws, reflect):
        self.target = target
        self.kernel = Gaussian(ms, Vs)
        self.reflect = reflect
        if ndraws == None:
            ndraws = self.kernel.param_dim
            warnings.warn('Setting ndraws to the minimum number: %d' % ndraws)

        # Sample random points
        t0 = time()
        self._sample(ndraws)
        self.time = time() - t0

    def _sample(self, ndraws):
        """
        Draw values from the specified kernel, compute sampled
        distribution values and importance weights.
        """
        self.x = self.kernel.sample(ndraws=ndraws)
        if self.reflect:
            self.x = reflect_sample(self.x)
        self.p = self.target(self.x)


