from time import time
from warnings import warn
import numpy as np

from .numlib import inv_sym_matrix
from .sampling import Sample
from .gaussian import (GaussianFamily,
                       FactorGaussianFamily)


families = {'gaussian': GaussianFamily,
            'factor_gaussian': FactorGaussianFamily}


class ImportanceFit(object):

    def __init__(self, target, sample, family='gaussian'):
        """
        Naive variational sampler object.
        """
        self._init_from_sample(target, sample, family)

    def _init_from_sample(self, target, sample, family):
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
        self._cache = {'F': self.family.design_matrix(sample.x)}

        # Perform fit
        self._do_fitting()
        self.time = time() - t0

    def _do_fitting(self):
        F = self._cache['F']
        p = self.target(self.sample.x).squeeze()
        if not self.sample.w is None:
            p *= self.sample.w
        moment = np.dot(F, p) / self.npts
        self._loc_fit = self.family.from_moment(moment)
        self._set_theta()
        # Compute variance on moment estimate
        if moment.ndim == 1:
            moment = np.reshape(moment, (moment.size, 1))
        self._var_moment = np.dot(F * (p ** 2), F.T) / self.npts\
            - np.dot(moment, moment.T)

    def _set_theta(self):
        if self.context is None:
            self._theta = self._loc_fit.theta
        elif self.family.check(self.context):
            self._theta = self._loc_fit.theta - self.context.theta
        else:
            try:
                fit = self._loc_fit / self.context
                self._theta = fit.theta
            except:
                self._theta = None

    def _get_theta(self):
        return self._theta

    def _get_fit(self):
        if self._theta is None:
            warn('cannot divide fit with context')
            return None
        return self.family.from_theta(self.theta)

    def _get_loc_fit(self):
        return self._loc_fit

    def _get_var_moment(self):
        return self._var_moment

    def _get_fisher_info(self):
        F = self._cache['F']
        q = np.nan_to_num(np.exp(np.dot(F.T, np.nan_to_num(self.theta))))
        return np.dot(F * q, F.T) / self.npts

    def _get_var_theta(self):
        inv_fisher_info = inv_sym_matrix(self.fisher_info)
        return np.dot(np.dot(inv_fisher_info, self._var_moment), inv_fisher_info)

    def _get_kl_error(self):
        return np.trace(self.var_moment * inv_sym_matrix(self.fisher_info))\
            / (2 * self.npts)

    theta = property(_get_theta)
    fit = property(_get_fit)
    loc_fit = property(_get_loc_fit)
    var_moment = property(_get_var_moment)
    var_theta = property(_get_var_theta)
    fisher_info = property(_get_fisher_info)
    kl_error = property(_get_kl_error)


class ImportanceSampler(ImportanceFit):  
    def __init__(self, target, kernel, context=None, ndraws=None, reflect=False,
                 family='gaussian'):
        S = Sample(kernel, context=context, ndraws=ndraws, reflect=reflect)
        self._init_from_sample(target, S, family)

