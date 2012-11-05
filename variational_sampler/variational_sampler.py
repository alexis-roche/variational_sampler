from time import time
from warnings import warn
import numpy as np

from .numlib import (inv_sym_matrix,
                     min_methods)
from .sampling import Sample
from .gaussian import (GaussianFamily,
                       FactorGaussianFamily)


families = {'gaussian': GaussianFamily,
            'factor_gaussian': FactorGaussianFamily}


class VariationalFit(object):
    
    def __init__(self, target, sample, family='gaussian',
                 theta=None, tol=1e-5, maxiter=None,
                 minimizer='newton'):
        """
        Variational sampler object.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : int
          Maximum number of iterations in optimization

        minimizer : string
          One of 'newton', 'quasi_newton', steepest', 'conjugate'
        """
        self._init_from_sample(target, sample, family, theta, tol, maxiter, minimizer)

    def _init_from_sample(self, target, sample, family, theta, tol, maxiter, minimizer):
        """
        Init object given a sample instance.
        """
        t0 = time()
        self.target = target
        self.sample = sample
        self.context = sample.context
        self.dim = sample.x.shape[0]
        self.npts = sample.x.shape[1]

        # Instantiate fitting family
        if family not in families.keys():
            raise ValueError('unknown family')
        self.family = families[family](self.dim)
        p = self.target(sample.x).squeeze()
        if not sample.w is None:
            p *= sample.w
        self._cache = {
            'theta': None,
            'F': self.family.design_matrix(sample.x),
            'p': p,
            'log_p': np.nan_to_num(np.log(p)),
            'q': np.zeros(p.size),
            'log_q': np.zeros(p.size)}

        # Initial guess
        if theta is None:
            self._theta = np.zeros(self._cache['F'].shape[0])
            self._theta[-1] = np.nan_to_num(np.log(np.sum(p) / np.sum(sample.w)))
        else:
            self._theta = np.asarray(theta)

        # Perform fitting
        self.minimizer = minimizer
        self.tol = tol
        self.maxiter = maxiter
        self._do_fitting()
        self.time = time() - t0

    def _udpate_fit(self, theta):
        """
        Compute fit
        """
        c = self._cache
        if not theta is c['theta']:
            c['log_q'][:] = np.dot(c['F'].T, np.nan_to_num(theta))
            c['q'][:] = np.nan_to_num(np.exp(c['log_q']))
            if not self.sample.w is None:
                c['q'] *= self.sample.w
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

    def _var_moment(self, theta):
        c = self._cache
        return np.dot(c['F'] * ((c['p'] - c['q']) ** 2), c['F'].T)\
            / self.npts

    def _fisher_info(self, theta):
        return self._hessian(self.theta) / self.npts

    def _do_fitting(self):
        """
        Perform Gaussian approximation.
        """
        theta = self._theta
        meth = self.minimizer
        if meth not in min_methods.keys():
            raise ValueError('unknown minimizer')
        if meth in ('newton', 'ncg'):
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  self._hessian,
                                  maxiter=self.maxiter, tol=self.tol)
        elif meth in ('quasi_newton',):
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  self._pseudo_hessian(),
                                  maxiter=self.maxiter, tol=self.tol)
        else:
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  maxiter=self.maxiter, tol=self.tol)
        m.message()
        self._theta = m.argmin()
        self.minimizer = m

    def _get_theta(self):
        return self._theta

    def _get_fit(self):
        return self.family.from_theta(self.theta)

    def _get_loc_fit(self):
        if self.context is None:
            return self._get_fit()
        elif self.family.check(self.context):
            return self.family.from_theta(self.theta + self.context.theta)
        else:
            try:
                return self._get_fit() * self.context
            except:
                warn('cannnot multiply fit with context')
                return None

    def _get_var_moment(self):
        return self._var_moment(self.theta)

    def _get_fisher_info(self):
        return self._fisher_info(self.theta)

    def _get_var_theta(self):
        inv_fisher_info = inv_sym_matrix(self.fisher_info)
        return np.dot(np.dot(inv_fisher_info, self._var_moment(self.theta)),
                      inv_fisher_info)

    def _get_kl_error(self):
        """
        Estimate the expected excess KL divergence.
        """
        return np.trace(self.var_moment * inv_sym_matrix(self.fisher_info))\
            / (2 * self.npts)

    theta = property(_get_theta)
    fit = property(_get_fit)
    loc_fit = property(_get_loc_fit)
    var_moment = property(_get_var_moment)
    var_theta = property(_get_var_theta)
    fisher_info = property(_get_fisher_info)
    kl_error = property(_get_kl_error)


class VariationalSampler(VariationalFit):
    def __init__(self, target, kernel, context=None, ndraws=None, reflect=False,
                 family='gaussian', theta=None, tol=1e-5, maxiter=None, minimizer='newton'):
        S = Sample(kernel, context=context, ndraws=ndraws, reflect=reflect)
        self._init_from_sample(target, S, family, theta, tol, maxiter, minimizer)
