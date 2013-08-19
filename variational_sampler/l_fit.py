from time import time
import numpy as np

from .numlib import inv_sym_matrix
from .gaussian import (Gaussian,
                       GaussianFamily,
                       FactorGaussianFamily)


families = {'gaussian': GaussianFamily,
            'factor_gaussian': FactorGaussianFamily}


class LFit(object):

    def __init__(self, sample, family='gaussian'):
        """
        Importance weighted likelihood fitting method.
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
        self._cache = {'F': self.family.design_matrix(sample.x)}

        # Perform fit
        self._do_fitting()
        self.time = time() - t0

    def _do_fitting(self):
        F = self._cache['F']
        self._integral = np.dot(F, self.sample.pe) / self.npts
        self._integral *= np.exp(self.sample.logscale)
        self._fit = self.family.from_integral(self._integral)

    def _get_integral(self):
        return self._integral

    def _get_var_integral(self):
        """
        Estimate variance on integral estimate
        """
        F, pe, integral = self._cache['F'], self.sample.pe, self._integral
        n = integral.size
        var = np.dot(F * (pe ** 2), F.T) / self.npts \
            - np.dot(integral.reshape(n, 1), integral.reshape(1, n))
        var /= self.npts
        return var

    def _get_fit(self):
        return self._fit

    def _get_theta(self):
        theta = self._fit.theta.copy()
        theta -= self.sample.kernel.theta
        return theta

    def _get_sensitivity_matrix(self):
        F = self._cache['F']
        # compute the fitted importance weights
        log_qe = np.dot(F.T, self._fit.theta) +\
            - self.sample.kernel.log(self.sample.x)
        qe = np.exp(log_qe - self.sample.logscale)
        return np.dot(F * qe, F.T) *\
            (np.exp(self.sample.logscale) / self.npts)

    def _get_var_theta(self):
        inv_sensitivity_matrix = inv_sym_matrix(self.sensitivity_matrix)
        return np.dot(np.dot(inv_sensitivity_matrix, self.var_integral),
                      inv_sensitivity_matrix)

    def _get_kl_error(self):
        return .5 * np.trace(np.dot(self.var_integral,
                                    inv_sym_matrix(self.sensitivity_matrix)))

    theta = property(_get_theta)
    fit = property(_get_fit)
    integral = property(_get_integral)
    var_integral = property(_get_var_integral)
    var_theta = property(_get_var_theta)
    sensitivity_matrix = property(_get_sensitivity_matrix)
    kl_error = property(_get_kl_error)
