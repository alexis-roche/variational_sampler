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
        self._moment = np.dot(F, self.sample.pe) / self.npts
        self._fit = self.family.from_moment(self._moment)

    def _get_moment(self):
        return np.exp(self.sample.logscale) * self._moment

    def _get_var_moment(self):
        """
        Estimate variance on moment estimate
        """
        F, pe, moment = self._cache['F'], self.sample.pe, self._moment
        n = moment.size
        var = np.dot(F * (pe ** 2), F.T) / self.npts \
            - np.dot(moment.reshape(n, 1), moment.reshape(1, n))
        var *= np.exp(2 * self.sample.logscale) / self.npts
        return var

    def _get_fit(self):
        K = self._fit.K * np.exp(self.sample.logscale)
        return Gaussian(self._fit.m, self._fit.V, K)

    def _get_theta(self):
        theta = self._fit.theta.copy()
        theta[0] += self.sample.logscale
        theta -= self.sample.kernel.theta
        return theta

    def _get_fisher_info(self):
        F = self._cache['F']
        # compute the fitted importance weights
        log_qe = np.dot(F.T, self._fit.theta) + self.sample.logscale \
            - self.sample.kernel.log(self.sample.x)
        qe = np.exp(log_qe)
        return np.dot(F * qe, F.T) \
            * (np.exp(self.sample.logscale) / self.npts)

    def _get_var_theta(self):
        inv_fisher_info = inv_sym_matrix(self.fisher_info)
        return np.dot(np.dot(inv_fisher_info, self.var_moment),
                      inv_fisher_info)

    def _get_kl_error(self):
        return .5 * np.trace(np.dot(self.var_moment,
                                    inv_sym_matrix(self.fisher_info)))

    theta = property(_get_theta)
    fit = property(_get_fit)
    moment = property(_get_moment)
    var_moment = property(_get_var_moment)
    var_theta = property(_get_var_theta)
    fisher_info = property(_get_fisher_info)
    kl_error = property(_get_kl_error)
