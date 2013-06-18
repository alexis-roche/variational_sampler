from time import time
import numpy as np

from .numlib import inv_sym_matrix, min_methods
from .gaussian import GaussianFamily, FactorGaussianFamily

VERBOSE = True

families = {'gaussian': GaussianFamily,
            'factor_gaussian': FactorGaussianFamily}


class KLFit(object):

    def __init__(self, sample, family='gaussian', tol=1e-5, maxiter=None,
                 minimizer='newton'):
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
        self._cache = {
            'theta': None,
            'F': self.family.design_matrix(sample.x),
            'qe': None,
            'log_qe': None
            }

        # Initial guess for theta parameter (default is optimal constant fit)
        self._theta_init = np.zeros(self._cache['F'].shape[0])
        if self.sample.Z is None:
            self._theta_init[0] = np.log(np.mean(self.sample.pe))
        else:
            self._theta_init[0] = np.log(self.sample.Z) - self.sample.logscale

        # Perform fit
        self.minimizer = minimizer
        self.tol = tol
        self.maxiter = maxiter
        self._do_fitting()
        self.time = time() - t0

    def _update_fit(self, theta):
        """
        Compute fit
        """
        c = self._cache
        if not theta is c['theta']:
            c['log_qe'] = np.dot(c['F'].T, theta)
            c['qe'] = np.exp(c['log_qe'])
            c['theta'] = theta
            fail = np.isinf(c['log_qe']).max() or np.isinf(c['qe']).max()
        else:
            fail = False
        return not fail

    def _loss(self, theta):
        """
        Compute the empirical divergence:

          sum[pe * log pe/qe + qe - pe],

        where:
          pe is the target distribution
          qe is the parametric fit
        """
        if not self._update_fit(theta):
            return np.inf
        c = self._cache
        return np.sum(self.sample.pe * (self.sample.log_pe - c['log_qe'])
                      + c['qe'] - self.sample.pe)

    def _gradient(self, theta):
        """
        Compute the gradient of the loss.
        """
        self._update_fit(theta)
        c = self._cache
        return np.dot(c['F'], c['qe'] - self.sample.pe)

    def _hessian(self, theta):
        """
        Compute the hessian of the loss.
        """
        self._update_fit(theta)
        c = self._cache
        return np.dot(c['F'] * c['qe'], c['F'].T)

    def _pseudo_hessian(self):
        """
        Approximate the Hessian at the minimum by substituting the
        fitted distribution with the target distribution.
        """
        c = self._cache
        return np.dot(c['F'] * self.sample.pe, c['F'].T)

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

    def _var_integral(self, theta):
        self._update_fit(theta)
        c = self._cache
        return np.dot(c['F'] * ((self.sample.pe - c['qe']) ** 2), c['F'].T)\
            * (np.exp(2 * self.sample.logscale) / (self.npts ** 2))

    def _sensitivity_matrix(self, theta):
        return self._hessian(self._theta) *\
            (np.exp(self.sample.logscale) / self.npts)

    def _get_theta(self):
        theta = self._theta.copy()
        theta[0] += self.sample.logscale
        return theta

    def _get_fit(self):
        if self.family.check(self.sample.kernel):
            return self.family.from_theta(\
                self.theta + self.sample.kernel.theta)
        elif self.sample.kernel.theta_dim < self.family.theta_dim:
            kernel = self.sample.kernel.embed()
            return self.family.from_theta(kernel.theta + self.theta)
        else:
            return self.family.from_theta(self.theta) * self.sample.kernel

    def _get_var_integral(self):
        return self._var_integral(self._theta)

    def _get_sensitivity_matrix(self):
        return self._sensitivity_matrix(self._theta)

    def _get_var_theta(self):
        inv_sensitivity_matrix = inv_sym_matrix(self.sensitivity_matrix)
        return np.dot(np.dot(inv_sensitivity_matrix,
                             self._var_integral(self._theta)),
                      inv_sensitivity_matrix)

    def _get_kl_error(self):
        """
        Estimate the expected excess KL divergence.
        """
        return .5 * np.trace(np.dot(self.var_integral,
                                    inv_sym_matrix(self.sensitivity_matrix)))

    theta = property(_get_theta)
    fit = property(_get_fit)
    var_integral = property(_get_var_integral)
    var_theta = property(_get_var_theta)
    sensitivity_matrix = property(_get_sensitivity_matrix)
    kl_error = property(_get_kl_error)
