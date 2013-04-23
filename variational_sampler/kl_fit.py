from time import time
from warnings import warn
import numpy as np

from .numlib import safe_exp, inv_sym_matrix, min_methods
from .gaussian import GaussianFamily, FactorGaussianFamily


families = {'gaussian': GaussianFamily,
            'factor_gaussian': FactorGaussianFamily}


class KLFit(object):
    
    def __init__(self, sample, family='gaussian', tol=1e-5, maxiter=None,
                 minimizer='newton', theta=None):
        """
        Sampling-based KL divergence minimization.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : int
          Maximum number of iterations in optimization

        minimizer : string
          One of 'newton', 'quasi_newton', steepest', 'conjugate'
        """
        t0 = time()
        self.sample = sample
        self.dim = sample.x.shape[0]
        self.npts = sample.x.shape[1]

        # Instantiate fitting family
        if family not in families.keys():
            raise ValueError('unknown family')
        self.family = families[family](self.dim)
        pw, self.logscale = safe_exp(self.sample.log_p + self.sample.log_w)
        self._cache = {
            'theta': None,
            'F': self.family.design_matrix(sample.x),
            'pw': pw,
            'log_p': self.sample.log_p,
            'qw': None,
            'log_q': None
            }

        # Initial guess for theta parameter (default is optimal constant fit)
        if theta is None:
            self._theta_init = np.zeros(self._cache['F'].shape[0])
            tmp = np.sum(self._cache['pw'])
            tmp2 = np.sum(np.exp(self.sample.log_w - self.logscale))
            self._theta_init[0] = np.nan_to_num(np.log(tmp / tmp2))
        else:
            self._theta_init = np.asarray(theta)

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
            c['log_q'] = np.dot(c['F'].T, theta)
            c['qw'] = np.exp(c['log_q'] + self.sample.log_w - self.logscale)
            c['theta'] = theta
            fail = np.isinf(c['log_q']).max() or np.isinf(c['qw']).max()
        else:
            fail = False
        return not fail

    def _loss(self, theta):
        """
        Compute the empirical divergence:

          sum[pw * log pw/qw + qw - pw],

        where:
          pw is the target distribution
          qw is the parametric fit
        """
        if not self._udpate_fit(theta):
            return np.inf
        c = self._cache
        return np.sum(c['pw'] * (c['log_p'] - c['log_q'])
                      + c['qw'] - c['pw'])

    def _gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self._udpate_fit(theta)
        c = self._cache
        return np.dot(c['F'], c['qw'] - c['pw'])

    def _hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._udpate_fit(theta)
        c = self._cache
        return np.dot(c['F'] * c['qw'], c['F'].T)

    def _pseudo_hessian(self):
        """
        Approximate the Hessian at the minimum by substituting the
        fitted distribution with the target distribution.
        """
        c = self._cache
        return np.dot(c['F'] * c['pw'], c['F'].T)

    def _do_fitting(self):
        """
        Perform Gaussian approximation.
        """
        theta = self._theta_init
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

    def _var_moment(self, theta):
        c = self._cache
        return np.dot(c['F'] * ((c['pw'] - c['qw']) ** 2), c['F'].T)\
            * (np.exp(2 * self.logscale) / (self.npts ** 2))

    def _fisher_info(self, theta):
        return self._hessian(self.theta) * (np.exp(self.logscale) / self.npts)

    def _get_theta(self):
        return self._theta

    def _get_fit(self):
        return self.family.from_theta(self.theta)

    def _get_glob_fit(self):
        if self.sample.context is None:
            return self._get_fit()
        elif self.family.check(self.sample.context):
            return self.family.from_theta(self.theta + self.sample.context.theta)
        else:
            try:
                return self._get_fit() * self.sample.context
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
        return .5 * np.trace(self.var_moment * inv_sym_matrix(self.fisher_info))

    theta = property(_get_theta)
    fit = property(_get_fit)
    glob_fit = property(_get_glob_fit)
    var_moment = property(_get_var_moment)
    var_theta = property(_get_var_theta)
    fisher_info = property(_get_fisher_info)
    kl_error = property(_get_kl_error)
