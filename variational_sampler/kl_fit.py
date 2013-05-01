from time import time
from warnings import warn
import numpy as np

from .numlib import safe_exp, inv_sym_matrix, min_methods
from .gaussian import GaussianFamily, FactorGaussianFamily

VERBOSE = True

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

        # Pre-compute some stuff and cache it
        log_pw = self.sample.log_p + self.sample.log_w
        pw, self.logscale = safe_exp(log_pw)
        log_pw -= self.logscale
        self._cache = {
            'theta': None,
            'F': self.family.design_matrix(sample.x),
            'pw': pw,
            'log_pw': log_pw,
            'qw': None,
            'log_qw': None
            }

        # Initial guess for theta parameter (default is optimal constant fit)
        if theta is None:
            self._theta_init = np.zeros(self._cache['F'].shape[0])
            self._theta_init[0] = np.log(np.mean(self._cache['pw']))
        else:
            self._theta_init = np.asarray(theta)

        # Perform fit
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
            c['log_qw'] = np.dot(c['F'].T, theta)
            c['qw'] = np.exp(c['log_qw'])
            c['theta'] = theta
            fail = np.isinf(c['log_qw']).max() or np.isinf(c['qw']).max()
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
        return np.sum(c['pw'] * (c['log_pw'] - c['log_qw'])
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
                                  maxiter=self.maxiter, tol=self.tol,
                                  verbose=VERBOSE)
        elif meth in ('quasi_newton',):
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  self._pseudo_hessian(),
                                  maxiter=self.maxiter, tol=self.tol,
                                  verbose=VERBOSE)
        else:
            m = min_methods[meth](theta, self._loss, self._gradient,
                                  maxiter=self.maxiter, tol=self.tol,
                                  verbose=VERBOSE)
        if VERBOSE:
            m.message()
        self._theta = m.argmin()
        self.minimizer = m

    def _var_moment(self, theta):
        self._udpate_fit(theta)
        c = self._cache
        return np.dot(c['F'] * ((c['pw'] - c['qw']) ** 2), c['F'].T)\
            * (np.exp(2 * self.logscale) / (self.npts ** 2))

    def _fisher_info(self, theta):
        return self._hessian(self._theta) * (np.exp(self.logscale) / self.npts)

    def _get_theta(self):
        theta = self._theta.copy()
        theta[0] += self.logscale
        return theta

    def _get_fit(self):
        return self.family.from_theta(self.sample.kernel.theta + self.theta)

    def _get_var_moment(self):
        return self._var_moment(self._theta)

    def _get_fisher_info(self):
        return self._fisher_info(self._theta)

    def _get_var_theta(self):
        inv_fisher_info = inv_sym_matrix(self.fisher_info)
        return np.dot(np.dot(inv_fisher_info, self._var_moment(self._theta)),
                      inv_fisher_info)

    def _get_kl_error(self):
        """
        Estimate the expected excess KL divergence.
        """
        return .5 * np.trace(np.dot(self.var_moment, inv_sym_matrix(self.fisher_info)))

    theta = property(_get_theta)
    fit = property(_get_fit)
    var_moment = property(_get_var_moment)
    var_theta = property(_get_var_theta)
    fisher_info = property(_get_fisher_info)
    kl_error = property(_get_kl_error)
