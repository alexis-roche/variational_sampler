from time import time
from warnings import warn
import numpy as np

from .numlib import safe_exp, inv_sym_matrix
from .gaussian import (GaussianFamily,
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
        pw, self.logscale = safe_exp(self.sample.log_p + self.sample.log_w)
        self._cache = {'F': self.family.design_matrix(sample.x),
                       'pw': pw,
                       'moment': None}

        # Perform fit
        self._do_fitting()
        self.time = time() - t0

    def _do_fitting(self):
        F, pw = self._cache['F'], self._cache['pw']
        moment = np.dot(F, pw) / self.npts
        self._glob_fit = self.family.from_moment(moment)
        scale = np.exp(self.logscale)
        self._glob_fit.rescale(scale)
        self._set_theta()
        self._cache['moment'] = moment

    def _get_moment(self):
        return np.exp(self.logscale) * self._cache['moment']

    def _get_var_moment(self):
        """
        Estimate variance on moment estimate
        """
        F, pw, moment = self._cache['F'], self._cache['pw'], self._cache['moment']
        scale = np.exp(self.logscale)
        var = np.dot(F * (pw ** 2), F.T) / self.npts \
            - np.dot(moment.reshape(moment.size, 1), moment.reshape(1, moment.size))
        var *= (scale ** 2 / self.npts)
        return var

    def _set_theta(self):
        if self.sample.context is None:
            self._theta = self._glob_fit.theta
        elif self.family.check(self.sample.context):
            self._theta = self._glob_fit.theta - self.sample.context.theta
        else:
            try:
                fit = self._glob_fit / self.sample.context
                self._theta = fit.theta
            except:
                self._theta = None

    def _get_theta(self):
        return self._theta

    def _get_fit(self):
        if self.sample.context is None:
            return self._glob_fit
        elif self._theta is None:
            warn('cannot divide fit with context')
            return None
        return self.family.from_theta(self.theta)

    def _get_glob_fit(self):
        return self._glob_fit

    def _get_fisher_info(self):
        F = self._cache['F']
        q = np.nan_to_num(np.exp(np.dot(F.T, np.nan_to_num(self.theta))\
                                     + self.sample.log_w))
        return np.dot(F * q, F.T) / self.npts

    def _get_var_theta(self):
        inv_fisher_info = inv_sym_matrix(self.fisher_info)
        return np.dot(np.dot(inv_fisher_info, self.var_moment), inv_fisher_info)

    def _get_kl_error(self):
        return .5 * np.trace(self.var_moment * inv_sym_matrix(self.fisher_info))

    theta = property(_get_theta)
    fit = property(_get_fit)
    glob_fit = property(_get_glob_fit)
    moment = property(_get_moment)
    var_moment = property(_get_var_moment)
    var_theta = property(_get_var_theta)
    fisher_info = property(_get_fisher_info)
    kl_error = property(_get_kl_error)
