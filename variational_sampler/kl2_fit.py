from time import time
import numpy as np

from .numlib import inv_sym_matrix, SteepestDescent
from .gaussian import GaussianFamily, FactorGaussianFamily

VERBOSE = False

families = {'gaussian': GaussianFamily,
            'factor_gaussian': FactorGaussianFamily}


class KL2Fit(object):

    def __init__(self, sample, family='gaussian', tol=1e-5, maxiter=None):
        """
        Sampling-based KL divergence minimization.

        Parameters
        ----------
        tol : float
          Tolerance on optimized parameter

        maxiter : int
          Maximum number of iterations in optimization
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
            'tau': None,
            'F': self.family.design_matrix(sample.x),
            'hat_tau_p': None,
            'qe': None,
            'log_qe': None
            }

        # Perform fit
        self.tol = tol
        if maxiter is None:
            maxiter = np.inf
        self.maxiter = maxiter
        self._do_fitting()
        self.time = time() - t0

    def _update_cache(self, tau):
        if tau is self._cache['tau']:
            return
        self._cache['tau'] = tau
        q = self.family.from_integral(tau)
        self._cache['log_qe'] = q.log(self.sample.x) -\
            self.sample.kernel.log(self.sample.x)
        self._cache['qe'] = np.exp(self._cache['log_qe'])

    def _loss(self, tau):
        self._update_cache(tau)
        c = self._cache
        return np.sum(self.sample.pe * (self.sample.log_pe - c['log_qe'])
                      + c['qe'] - self.sample.pe)

    def _pseudo_gradient(self, tau):
        self._update_cache(tau)
        c = self._cache
        hat_tau = np.dot(c['F'], self._cache['qe']) / self.npts
        return hat_tau - c['hat_tau_p']

    def _do_fitting(self):
        """
        Perform Gaussian approximation.
        """
        # Compute importance sampling integral estimate
        F = self._cache['F']
        self._cache['hat_tau_p'] = np.dot(F, self.sample.pe) / self.npts
        # Define the initial fit as the (unnormalized) sampling kernel
        if self.family.check(self.sample.kernel):
            q = self.sample.kernel
        elif self.sample.kernel.theta_dim < self.family.theta_dim:
            q = self.sample.kernel.embed()
        else:
            raise ValueError('inconsistent sampling kernel and fitting family')
        if self.sample.Z is None:
            Z = np.mean(self.sample.pe)
        else:
            Z = self.sample.Z / np.exp(self.sample.logscale)
        q.rescale(Z)
        tau = q.integral()
        m = SteepestDescent(tau, self._loss, self._pseudo_gradient,
                            maxiter=self.maxiter, tol=self.tol,
                            verbose=VERBOSE)
        if VERBOSE:
            m.message()
        tau = m.argmin()
        self._update_cache(tau)
        self._integral = np.exp(self.sample.logscale) * tau
        self._fit = self.family.from_integral(self._integral)

    def _get_var_integral(self):
        c = self._cache
        return np.dot(c['F'] * ((self.sample.pe - c['qe']) ** 2), c['F'].T) *\
            (np.exp(2 * self.sample.logscale) / (self.npts ** 2))

    def _get_fit(self):
        return self._fit

    def _get_theta(self):
        theta = self._fit.theta.copy()
        theta -= self.sample.kernel.theta
        return theta

    def _get_sensitivity_matrix(self):
        c = self._cache
        return np.dot(c['F'] * c['qe'], c['F'].T) *\
            (np.exp(self.sample.logscale) / self.npts)

    def _get_var_theta(self):
        inv_sensitivity_matrix = inv_sym_matrix(self.sensitivity_matrix)
        return np.dot(np.dot(inv_sensitivity_matrix, self.var_integral),
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
