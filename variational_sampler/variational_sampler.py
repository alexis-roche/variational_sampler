from time import time
import numpy as np

from .numlib import (SteepestDescent,
                     ConjugateDescent,
                     NewtonDescent,
                     QuasiNewtonDescent,
                     ScipyCG, ScipyNCG, ScipyBFGS)
from .sampling import Sample
from .gaussian import Gaussian

MIN_IMPL = {'steepest': SteepestDescent,
            'conjugate': ConjugateDescent,
            'newton': NewtonDescent,
            'quasi_newton': QuasiNewtonDescent,
            'cg': ScipyCG,
            'ncg': ScipyNCG,
            'bfgs': ScipyBFGS
            }


def make_design(x):
    """
    Assemble design matrix with general term: F_ik = phi_i(x_k)
    """
    I, J = np.triu_indices(x.shape[0])
    F = np.array([x[i, :] * x[j, :] for i, j in zip(I, J)])
    return np.concatenate((F, x, np.ones([1, x.shape[1]])))


class DirectFit(object):

    def __init__(self, S):
        """
        Naive variational sampler object.
        """
        self._init_from_sample(S)

    def _init_from_sample(self, S):
        t0 = time()
        self.sample = S
        self.dim = S.x.shape[0]
        self.npts = S.x.shape[1]
        
        # Make a cache: pre-compute the design matrix
        self._cache = {'F': make_design(S.x)}

        # Perform fit
        self._do_fitting()
        self.time = time() - t0

    def _do_fitting(self):
        F = self._cache['F']
        p = self.sample.p
        moment = np.dot(F, p) / self.npts
        Z = moment[-1]
        m = moment[-1 - self.dim:-1] / Z
        V = np.zeros((self.dim, self.dim))
        idx = np.triu_indices(self.dim) 
        V[idx] = moment[0:-1 - self.dim] / Z
        V[np.tril_indices(self.dim)] = V[idx]
        V -= np.diag(m ** 2)
        self._loc_fit = Gaussian(m, V, Z=Z)
        if moment.ndim == 1:
            moment = np.reshape(moment, (moment.size, 1))
        self._sigma1 = np.dot(F * (p ** 2), F.T) / self.npts\
            - np.dot(moment, moment.T)

    def _get_theta(self):
        return self._loc_fit.theta - self.sample.kernel.theta

    def _get_fit(self):
        return Gaussian(theta=self.theta)

    def _get_loc_fit(self):
        return self._loc_fit

    def _get_sigma1(self):
        return self._sigma1

    def _get_sigma2(self):
        F = self._cache['F']
        q = np.nan_to_num(np.exp(np.dot(F.T, np.nan_to_num(self.theta))))
        return np.dot(F * q, F.T) / self.npts

    def _get_kl_error(self):
        return np.trace(self.sigma1 * np.linalg.inv(self.sigma2))\
            / (2 * self.npts)

    theta = property(_get_theta)
    fit = property(_get_fit)
    loc_fit = property(_get_loc_fit)
    sigma1 = property(_get_sigma1)
    sigma2 = property(_get_sigma2)
    kl_error = property(_get_kl_error)


class VariationalFit(object):
    
    def __init__(self, S, tol=1e-7, maxiter=None, minimizer='newton'):
        """
        Variational sampler object.
        """
        self._init_from_sample(S, tol, maxiter, minimizer)

    def _init_from_sample(self, S, tol, maxiter, minimizer):
        """
        Init object given a sample instance.
        """
        t0 = time()
        self.sample = S
        self.dim = S.x.shape[0]
        self.npts = S.x.shape[1]

        # Make a cache: pre-allocate various arrays and pre-compute
        # the design matrix
        self._cache = {
            'theta': None,
            'F': make_design(S.x),
            'p': S.p,
            'log_p': np.nan_to_num(np.log(S.p)),
            'q': np.zeros(S.p.size),
            'log_q': np.zeros(S.p.size)}

        # Perform fitting
        self._do_fitting(tol, maxiter, minimizer)
        self.time = time() - t0

    def _udpate_fit(self, theta):
        """
        Compute fit
        """
        c = self._cache
        if not theta is c['theta']:
            c['log_q'][:] = np.dot(c['F'].T, np.nan_to_num(theta))
            c['q'][:] = np.nan_to_num(np.exp(c['log_q']))
            c['theta'] = theta

    def _loss(self, theta):
        """
        Compute the empirical divergence:

          sum[p * log p/q + q - p],

        where:
          p is the target distribution
          q is the parametric fit
        """
        self._udpate_fit(theta)
        c = self._cache
        return np.sum(c['p'] * (c['log_p'] - c['log_q'])
                      + c['q'] - c['p'])

    def _gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self._udpate_fit(theta)
        c = self._cache
        return np.dot(c['F'], c['q'] - c['p'])

    def _hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._udpate_fit(theta)
        c = self._cache
        return np.dot(c['F'] * c['q'], c['F'].T)

    def _pseudo_hessian(self):
        """
        Approximate the Hessian at the minimum by substituting the
        fitted distribution with the target distribution.
        """
        c = self._cache
        return np.dot(c['F'] * c['p'], c['F'].T)

    def _sigma1(self, theta):
        c = self._cache
        return np.dot(c['F'] * ((c['p'] - c['q']) ** 2), c['F'].T)\
            / self.npts

    def _sigma2(self, theta):
        return self._hessian(self.theta) / self.npts

    def _do_fitting(self, tol, maxiter, minimizer):
        """
        Perform Gaussian approximation.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : int
          Maximum number of iterations in optimization

        minimizer : string
          One of 'steepest', 'conjugate', 'newton'

        Returns
        -------
        fit : Gaussian object
          Gaussian fit
        """
        theta = np.zeros(self._cache['F'].shape[0])
        if minimizer not in MIN_IMPL.keys():
            raise ValueError('unknown minimizer')
        if minimizer in ('newton', 'ncg'):
            m = MIN_IMPL[minimizer](theta, self._loss, self._gradient,
                                    self._hessian, maxiter=maxiter, tol=tol)
        elif minimizer in ('quasi_newton',):
            m = MIN_IMPL[minimizer](theta, self._loss, self._gradient,
                                    self._pseudo_hessian(), maxiter=maxiter, tol=tol)
        else:
            m = MIN_IMPL[minimizer](theta, self._loss, self._gradient,
                                    maxiter=maxiter, tol=tol)
        m.message()
        self._theta = m.argmin()
        self.minimizer = m

    def _get_theta(self):
        return self._theta

    def _get_fit(self):
        return Gaussian(theta=self.theta)

    def _get_loc_fit(self):
        return Gaussian(theta=self.theta
                        + self.sample.kernel.theta)

    def _get_sigma1(self):
        return self._sigma1(self.theta)

    def _get_sigma2(self):
        return self._sigma2(self.theta)

    def _get_kl_error(self):
        """
        Estimate the expected excess KL divergence.
        """
        return np.trace(self.sigma1 * np.linalg.inv(self.sigma2))\
            / (2 * self.npts)

    theta = property(_get_theta)
    fit = property(_get_fit)
    loc_fit = property(_get_loc_fit)
    sigma1 = property(_get_sigma1)
    sigma2 = property(_get_sigma2)
    kl_error = property(_get_kl_error)


class DirectSampler(DirectFit):    
    def __init__(self, target, ms, Vs, ndraws=None, reflect=False):
        S = Sample(target, ms, Vs,
                   ndraws=ndraws,
                   reflect=reflect)
        self._init_from_sample(S)


class VariationalSampler(VariationalFit):
    def __init__(self, target, ms, Vs, ndraws=None, reflect=False,
                 tol=1e-7, maxiter=None, minimizer='newton'):
        S = Sample(target, ms, Vs,
                   ndraws=ndraws,
                   reflect=reflect)
        self._init_from_sample(S, tol, maxiter, minimizer)
