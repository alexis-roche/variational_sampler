import numpy as np

from .numlib import (force_tiny, minimize)
from .sampling import Sample
from .gaussian import Gaussian


def make_design(x):
    """
    Assemble design matrix with general term: F_ik = phi_i(x_k)
    """
    I, J = np.triu_indices(x.shape[0])
    F = np.array([x[i, :] * x[j, :] for i, j in zip(I, J)])
    return np.concatenate((F, x, np.ones([1, x.shape[1]])))


def kl_error(sigma1, sigma2, N):
    """
    Estimate the expected error on the excess KL divergence.
    """
    return np.trace(sigma1 * np.linalg.inv(sigma2)) / (2 * N)


class VariationalSampler(object):
    
    def __init__(self, target, ms, Vs, ndraws=None, reflect=False):
        """
        Variational importance sampling constructor
        """
        # Sampling step
        S = Sample(target, ms, Vs, ndraws=ndraws, reflect=reflect)
        self._sample = S
        self.dim = S.x.shape[0]
        self.kernel_theta = S.kernel.theta
        self.cache = {'theta': None}

        # Pre-compute design matrix
        self.F = make_design(S.x)

        # Pre-compute target distribution including zero-checking
        self.p = force_tiny(S.p)
        self.log_p = np.log(self.p)

        # Allocate arrays to deal with fitting
        self.q = np.zeros(self.p.size)
        self.log_q = np.zeros(self.p.size)
        self.theta = None

    def update_fit(self, theta):
        """
        Compute fit
        """
        if not theta is self.cache['theta']:
            self.log_q[:] = np.dot(self.F.T, np.nan_to_num(theta))
            self.q[:] = np.nan_to_num(np.exp(self.log_q))
            self.cache['theta'] = theta

    def loss(self, theta):
        """
        Compute the empirical divergence:

          sum[p * log p/q + q - p],

        where:
          p is the target distribution
          q is the parametric fit
        """
        self.update_fit(theta)
        return np.sum(self.p * (self.log_p - self.log_q)
                      + self.q - self.p)

    def gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self.update_fit(theta)
        return np.dot(self.F, self.q - self.p)

    def hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self.update_fit(theta)
        return np.dot(self.F * self.q, self.F.T)

    def sigma1(self):
        if self.theta == None:
            return None
        self.update_fit(self.theta)
        return np.dot(self.F * ((self.p - self.q) ** 2), self.F.T)\
            / self.F.shape[1]

    def sigma2(self):
        if self.theta == None:
            return None
        return self.hessian(self.theta) / self.F.shape[1]

    def kl_error(self):
        """
        Estimate the expected error on the excess KL divergence.
        """
        return kl_error(self.sigma1(), self.sigma2(), self.F.shape[1])

    def fit(self, maxiter=None, minimizer='cg'):
        """
        Perform Gaussian approximation.

        Parameters
        ----------
        maxiter : int
          Maximum number of iterations in the optimization

        minimizer : string
          One of 'newton', 'ncg', 'cg', 'bfgs'

        Returns
        -------
        fit : Gaussian object
          Gaussian fit
        """
        def callback(theta):
            print(theta)
        theta = np.zeros(self.F.shape[0])
        self.theta = minimize(self.loss,
                              theta,
                              self.gradient,
                              hess=self.hessian,
                              minimizer=minimizer,
                              maxiter=maxiter)
        return Gaussian(theta=self.theta)
