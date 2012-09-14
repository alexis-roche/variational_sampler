from time import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .gaussian import FactorGaussian
from .gaussian_mixture import GaussianMixture


class VariationalFitBMC(object):

    def __init__(self, S, var=1, damping=1e-5):
        """
        Bayesian quadrature using Gaussian kernels (Bayesian Monte
        Carlo method).
        
        p / pi approx Gauss mixture
        
        Parameters
        ----------
        S : Sample object
          Input sample
        
        var : float or array
          Isotropic or component-wise squared kernel size

        maxiter : int
          Maximum number of iterations for scale optimization

        damping : float
          Damping factor used for regularization to avoid ill-conditioning
        """
        self._init_from_sample(S, var, damping)

    def _init_from_sample(self, S, var, damping):
        t0 = time()
        self.sample = S
        self.dim = S.x.shape[0]
        self.npts = S.x.shape[1]
        var = np.asarray(var)
        if var.size == 1:
            self._v = np.zeros(self.dim)
            self._v.fill(var)
        else:
            self._v = np.reshape(var, (self.dim,))
        self.damping = float(damping)
        self._do_fitting()
        self.time = time() - t0
        
    def _do_fitting(self):
        # Assemble covariance matrix
        x = self.sample.x
        G = np.zeros((self.npts, self.npts))
        g = FactorGaussian(np.zeros(self.dim), self._v, K=1)
        for i in range(self.npts):
            G[i, :] = g((x.T[i] - x.T).T)
            G[i, i] += self.damping

        # Solve for the spline coefficients and compute model evidence
        L, lower = cho_factor(G, lower=0)
        self._theta = cho_solve((L, 0), self.sample.p)

    def _get_theta(self):
        return self._theta

    def _get_fit(self):
        x = self.sample.x
        gaussians = [FactorGaussian(xi, self._v, K=1) for xi in x.T]
        return GaussianMixture(self._theta, gaussians)

    def _get_loc_fit(self):
        x = self.sample.x
        gaussians = [self.sample.kernel * FactorGaussian(xi, self._v, K=1) for xi in x.T]
        return GaussianMixture(self._theta, gaussians)

    theta = property(_get_theta)
    fit = property(_get_fit)
    loc_fit = property(_get_loc_fit)


class VariationalSamplerBMC(VariationalFitBMC):    
    def __init__(self, target, ms, Vs, ndraws=None, reflect=False,
                 var=1, damping=1e-5):
        S = Sample(target, ms, Vs,
                   ndraws=ndraws,
                   reflect=reflect)
        self._init_from_sample(S, var, damping)
