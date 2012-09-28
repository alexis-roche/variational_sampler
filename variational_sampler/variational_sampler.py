from time import time
import numpy as np

from .numlib import (inv_sym_matrix,
                     SteepestDescent,
                     ConjugateDescent,
                     NewtonDescent,
                     QuasiNewtonDescent,
                     ScipyCG, ScipyNCG, ScipyBFGS)
from .sampling import Sample
from .gaussian import (GaussianFamily,
                       FactorGaussianFamily)

MINIMIZER_IMPL = {'steepest': SteepestDescent,
                  'conjugate': ConjugateDescent,
                  'newton': NewtonDescent,
                  'quasi_newton': QuasiNewtonDescent,
                  'cg': ScipyCG,
                  'ncg': ScipyNCG,
                  'bfgs': ScipyBFGS}

FAMILY_IMPL = {'gaussian': GaussianFamily,
               'factor_gaussian': FactorGaussianFamily}


class VariationalFit(object):
    
    def __init__(self, target, sample, family='gaussian',
                 tol=1e-5, maxiter=None, minimizer='newton'):
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
        self._init_from_sample(target, sample, family, tol, maxiter, minimizer)

    def _init_from_sample(self, target, sample, family, tol, maxiter, minimizer):
        """
        Init object given a sample instance.
        """
        t0 = time()
        self.target = target
        self.sample = sample
        self.dim = sample.x.shape[0]
        self.npts = sample.x.shape[1]

        # Instantiate fitting family
        if family not in FAMILY_IMPL.keys():
            raise ValueError('unknown family')
        self.family = FAMILY_IMPL[family](self.dim)
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
        theta = np.zeros(self._cache['F'].shape[0])
        minimizer = self.minimizer
        if minimizer not in MINIMIZER_IMPL.keys():
            raise ValueError('unknown minimizer')
        if minimizer in ('newton', 'ncg'):
            m = MINIMIZER_IMPL[minimizer](theta, self._loss, self._gradient,
                                          self._hessian,
                                          maxiter=self.maxiter, tol=self.tol)
        elif minimizer in ('quasi_newton',):
            m = MINIMIZER_IMPL[minimizer](theta, self._loss, self._gradient,
                                          self._pseudo_hessian(),
                                          maxiter=self.maxiter, tol=self.tol)
        else:
            m = MINIMIZER_IMPL[minimizer](theta, self._loss, self._gradient,
                                          maxiter=self.maxiter, tol=self.tol)
        m.message()
        self._theta = m.argmin()
        self.minimizer = m

    def _get_theta(self):
        return self._theta

    def _get_fit(self):
        return self.family.from_theta(theta=self.theta)

    def _get_loc_fit(self):
        return self.family.from_theta(theta=self.theta
                                      + self.sample.kernel.theta)

    def _get_var_moment(self):
        return self._var_moment(self.theta)

    def _get_fisher_info(self):
        return self._fisher_info(self.theta)

    def _get_var_theta(self):
        inv_fisher_info = inv_sym_matrix(self._fisher_info(self.theta))
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
    def __init__(self, target, kernel, generator=None, ndraws=None, reflect=False,
                 family='gaussian', tol=1e-5, maxiter=None, minimizer='newton'):
        S = Sample(kernel, generator=generator, ndraws=ndraws, reflect=reflect)
        self._init_from_sample(target, S, family, tol, maxiter, minimizer)


class StraightFit(object):

    def __init__(self, target, sample, family='gaussian'):
        """
        Naive variational sampler object.
        """
        self._init_from_sample(target, sample, family)

    def _init_from_sample(self, target, sample, family):
        t0 = time()
        self.target = target
        self.sample = sample
        self.dim = sample.x.shape[0]
        self.npts = sample.x.shape[1]
        
        # Instantiate fitting family
        if family not in FAMILY_IMPL.keys():
            raise ValueError('unknown family')
        self.family = FAMILY_IMPL[family](self.dim)
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
        # Compute variance on moment estimate
        if moment.ndim == 1:
            moment = np.reshape(moment, (moment.size, 1))
        self._var_moment = np.dot(F * (p ** 2), F.T) / self.npts\
            - np.dot(moment, moment.T)

    def _get_theta(self):
        return self._loc_fit.theta - self.sample.kernel.theta

    def _get_fit(self):
        return self.family.from_theta(theta=self.theta)

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


class ImportanceSampler(StraightFit):  
    def __init__(self, target, kernel, generator=None, ndraws=None, reflect=False,
                 family='gaussian'):
        S = Sample(kernel, generator=generator, ndraws=ndraws, reflect=reflect)
        self._init_from_sample(target, S, family)

