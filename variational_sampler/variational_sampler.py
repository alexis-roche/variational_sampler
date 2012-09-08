import numpy as np

from .numlib import minimize
from .sampling import Sample
from .gaussian import Gaussian


def make_design(x):
    """
    Assemble design matrix with general term: F_ik = phi_i(x_k)
    """
    I, J = np.triu_indices(x.shape[0])
    F = np.array([x[i, :] * x[j, :] for i, j in zip(I, J)])
    return np.concatenate((F, x, np.ones([1, x.shape[1]])))


class VariationalSampler(object):
    
    def __init__(self, target, ms, Vs, ndraws=None, reflect=False,
                 maxiter=None, minimizer='cg'):
        """
        Variational importance sampling object.
        """
        # Sampling step
        S = Sample(target, ms, Vs, ndraws=ndraws, reflect=reflect)
        self.sample = S
        self.dim = S.x.shape[0]
        self.cache = {'theta': None}

        # Pre-compute design matrix
        self.F = make_design(S.x)

        # Pre-compute target distribution including zero-checking
        self.p = S.p
        self.log_p = np.nan_to_num(np.log(self.p))

        # Allocate arrays to deal with fitting
        self.q = np.zeros(self.p.size)
        self.log_q = np.zeros(self.p.size)
        self.theta = None

        # Perform fitting
        self._fit(maxiter, minimizer)

    def _udpate_fit(self, theta):
        """
        Compute fit
        """
        if not theta is self.cache['theta']:
            self.log_q[:] = np.dot(self.F.T, np.nan_to_num(theta))
            self.q[:] = np.nan_to_num(np.exp(self.log_q))
            self.cache['theta'] = theta

    def _loss(self, theta):
        """
        Compute the empirical divergence:

          sum[p * log p/q + q - p],

        where:
          p is the target distribution
          q is the parametric fit
        """
        self._udpate_fit(theta)
        return np.sum(self.p * (self.log_p - self.log_q)
                      + self.q - self.p)

    def _gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self._udpate_fit(theta)
        return np.dot(self.F, self.q - self.p)

    def _hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._udpate_fit(theta)
        return np.dot(self.F * self.q, self.F.T)

    def sigma1(self):
        return np.dot(self.F * ((self.p - self.q) ** 2), self.F.T)\
            / self.F.shape[1]

    def sigma2(self):
        return self._hessian(self.theta) / self.F.shape[1]

    def kl_error(self):
        """
        Estimate the expected excess KL divergence.
        """
        return np.trace(self.sigma1() * np.linalg.inv(self.sigma2()))\
            / (2 * self.F.shape[1])

    def _fit(self, maxiter, minimizer):
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
        self.theta = minimize(self._loss,
                              theta,
                              self._gradient,
                              hess=self._hessian,
                              minimizer=minimizer,
                              maxiter=maxiter)

    def get_fit(self):
        return Gaussian(theta=self.theta)

    def get_local_fit(self):
        return Gaussian(theta=self.theta
                        + self.sample.kernel.theta)
