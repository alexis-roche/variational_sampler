"""
Constants used in several modules.
Basic implementation of Gauss-Newton gradient descent scheme.
"""
from time import time
from warnings import warn
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import fmin_cg, fmin_ncg, fmin_bfgs

dinfo = np.finfo(np.double)
TINY = dinfo.tiny
HUGE = dinfo.max


def force_tiny(x):
    return np.maximum(x, TINY)


def hdot(x, A):
    return np.dot(x, np.dot(A, x))


def safe_exp(x):
    xmax = x.max()
    return np.exp(x - xmax), xmax


def decomp_sym_matrix(A):
    s, P = eigh(A)
    sign_s = 2. * (s >= 0) - 1
    abs_s = force_tiny(np.abs(s))
    return abs_s, sign_s, P


def inv_sym_matrix(A):
    s, P = eigh(A)
    return np.dot(P * (1 / force_tiny(s)), P.T)


class SteepestDescent(object):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True, verbose=False):
        self._generic_init(x, f, grad_f, maxiter, tol,
                           stepsize, adaptive, verbose)
        self.run()

    def _generic_init(self, x, f, grad_f, maxiter, tol,
                      stepsize, adaptive, verbose):
        self.x = np.asarray(x).ravel()
        self.ref_norm = np.maximum(tol, np.max(np.abs(x)))
        self.f = f
        self.grad_f = grad_f
        if maxiter == None:
            maxiter = np.inf
        self.maxiter = maxiter
        self.tol = tol
        self.fval = self.f(self.x)
        self.fval0 = self.fval
        self.iter = 0
        self.nevals = 1
        self.a = stepsize
        self.adaptive = adaptive
        self.verbose = verbose

    def direction(self):
        return np.nan_to_num(-self.grad_f(self.x))

    def run(self):
        t0 = time()
        while self.iter < self.maxiter:
            # Evaluate function at current point
            xN = self.x
            fvalN = self.fval

            # Compute descent direction
            dx = self.direction()
            dx_norm = np.max(np.abs(dx))

            # Line search
            done = False
            stuck = False
            a = self.a
            while not done:
                x = np.nan_to_num(xN + a * dx)
                if not self.adaptive:
                    self.x = x
                    break
                fval = self.f(x)
                self.nevals += 1
                if fval < self.fval:
                    self.fval = fval
                    self.x = x
                    self.a = a
                    a *= 2
                else:
                    a *= .5
                    stuck = abs(a * dx_norm) < self.tol * self.ref_norm
                    done = self.fval < fvalN or stuck

            # Termination test
            self.iter += 1
            if self.verbose:
                print ('Iter:%d, f=%f, a=%f' % (self.iter, self.fval, self.a))
            if self.iter > self.maxiter or stuck:
                break

        self.time = time() - t0

    def argmin(self):
        return self.x

    def message(self):
        print('Number of iterations: %d' % self.iter)
        print('Number of function evaluations: %d' % self.nevals)
        print('Minimum criterion value: %f' % self.fval)
        print('Optimization time: %f' % self.time)


class ConjugateDescent(SteepestDescent):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True, verbose=False):
        self._generic_init(x, f, grad_f, maxiter, tol,
                           stepsize, adaptive, verbose)
        self.prev_dx = None
        self.prev_g = None
        self.run()

    def direction(self):
        """
        Polak-Ribiere rule. Reset direction if beta < 0 or if
        objective increases along proposed direction.
        """
        g = self.grad_f(self.x)
        if self.prev_dx == None:
            dx = -g
        else:
            b = max(0, np.dot(g, g - self.prev_g) / np.sum(self.prev_g ** 2))
            dx = -g + b * self.prev_dx
            if np.dot(dx, g) > 0:
                dx = -g
        self.prev_g = g
        self.prev_dx = dx
        return np.nan_to_num(dx)


class NewtonDescent(SteepestDescent):

    def __init__(self, x, f, grad_f, hess_f, maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True, verbose=False):
        self._generic_init(x, f, grad_f, maxiter, tol,
                           stepsize, adaptive, verbose)
        self.hess_f = hess_f
        self.run()

    def direction(self):
        """
        Compute the gradient g and Hessian H, then solve H dx = -g
        using the Cholesky decomposition: H = L L.T

        Upon failure, approximate the Hessian by a scalar matrix,
        i.e. H = tr(H) / n Id
        """
        g = self.grad_f(self.x)
        H = self.hess_f(self.x)
        try:
            L, _ = cho_factor(H, lower=0)
            dx = -cho_solve((L, 0), g)
        except:
            warn('Ooops... singular Hessian, regularizing')
            trH = force_tiny(np.trace(H))
            dx = -(H.shape[0] / trH) * g
        return np.nan_to_num(dx)


class QuasiNewtonDescent(SteepestDescent):

    def __init__(self, x, f, grad_f, fix_hess_f, maxiter=None, tol=1e-7,
                 stepsize=1., adaptive=True, verbose=False):
        self._generic_init(x, f, grad_f, maxiter, tol,
                           stepsize, adaptive, verbose)
        self.Hinv = inv_sym_matrix(fix_hess_f)
        self.run()

    def direction(self):
        g = self.grad_f(self.x)
        dx = -np.dot(self.Hinv, g)
        return np.nan_to_num(dx)


class ScipyCG(object):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7, verbose=False):
        t0 = time()
        stuff = fmin_cg(f, x, fprime=grad_f, args=(),
                        maxiter=maxiter, gtol=tol,
                        full_output=True, disp=verbose)
        self.x, self.fval = stuff[0], stuff[1]
        self.time = time() - t0

    def argmin(self):
        return self.x

    def message(self):
        print('Scipy conjugate gradient implementation')


class ScipyNCG(object):

    def __init__(self, x, f, grad_f, hess_f, maxiter=None, tol=1e-7,
                 verbose=False):
        t0 = time()
        stuff = fmin_ncg(f, x, grad_f, fhess=hess_f, args=(),
                         maxiter=maxiter, avextol=tol,
                         full_output=True, disp=verbose)
        self.x, self.fval = stuff[0], stuff[1]
        self.time = time() - t0

    def argmin(self):
        return self.x

    def message(self):
        print('Scipy Newton conjugate gradient implementation')


class ScipyBFGS(object):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7, verbose=False):
        t0 = time()
        stuff = fmin_bfgs(f, x, fprime=grad_f, args=(),
                          maxiter=maxiter, gtol=tol,
                          full_output=True, disp=verbose)
        self.x, self.fval = stuff[0], stuff[1]
        self.time = time() - t0

    def argmin(self):
        return self.x

    def message(self):
        print('Scipy BFGS quasi-Newton implementation')


min_methods = {'steepest': SteepestDescent,
               'conjugate': ConjugateDescent,
               'newton': NewtonDescent,
               'quasi_newton': QuasiNewtonDescent,
               'cg': ScipyCG,
               'ncg': ScipyNCG,
               'bfgs': ScipyBFGS}
