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


def safe_eigh(A):
    s, P = eigh(A)
    sign_s = 2. * (s >= 0) - 1
    abs_s = force_tiny(np.abs(s))
    return abs_s, sign_s, P


class SteepestDescent(object):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7):
        self._generic_init(x, f, grad_f, maxiter, tol)
        self.run()

    def _generic_init(self, x, f, grad_f, maxiter, tol):
        self.x = np.asarray(x).ravel()
        self.f = f
        self.grad_f = grad_f
        if maxiter == None:
            maxiter = np.inf
        self.maxiter = maxiter
        self.tol = tol
        self.fval = self.f(self.x)
        self.fval0 = self.fval
        self.iter = 0
        self.a = 1
        self.nevals = 1
        
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
            dx_inf = np.max(np.abs(dx))

            # Line search
            done = False
            stuck = False
            a = self.a
            while not done:
                x = xN + a * dx
                fval = self.f(x)
                self.nevals += 1
                if fval < self.fval:
                    self.fval = fval
                    self.x = x
                    self.a = a
                    a *= 2
                else:
                    a *= .5
                    stuck = abs(a * dx_inf) < self.tol
                    done = self.fval < fvalN or stuck

            # Termination test
            self.iter += 1
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

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7):
        self._generic_init(x, f, grad_f, maxiter, tol)
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

    def __init__(self, x, f, grad_f, hess_f, maxiter=None, tol=1e-7):
        self._generic_init(x, f, grad_f, maxiter, tol)
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

    def __init__(self, x, f, grad_f, fix_hess_f, maxiter=None, tol=1e-7):
        self._generic_init(x, f, grad_f, maxiter, tol)
        s, P = eigh(fix_hess_f)
        self.Hinv = np.dot(P * (1 / force_tiny(s)), P.T)
        self.run()
    
    def direction(self):
        g = self.grad_f(self.x)
        dx = -np.dot(self.Hinv, g)
        return np.nan_to_num(dx)


class BFGSQuasiNewtonDescent(SteepestDescent):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7):
        self._generic_init(x, f, grad_f, maxiter, tol)
        self.Hinv = np.eye(x.size)
        self.prev_g = None
        self.prev_dx = None
        self.run()
    
    def direction(self):
        """
        BFGS inverse Hessian approximation.
        """
        g = self.grad_f(self.x)
        if self.prev_dx == None:
            dx = -g
        else:
            s = self.a * self.prev_dx
            y = g - self.prev_g
            sy = np.dot(s, y)
            if abs(sy) > 1e-5:
                By = np.dot(self.Hinv, y)
                self.Hinv = self.Hinv + ((sy + np.dot(y, By)) / (sy ** 2)) * np.outer(s, s)\
                    - (1 / sy) * (np.outer(By, s) + np.outer(s, By))
            dx = -np.dot(self.Hinv, g)
        self.prev_g = g
        self.prev_dx = dx
        return np.nan_to_num(dx)


class ScipyCG(object):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7):
        t0 = time()
        stuff = fmin_cg(f, x, fprime=grad_f, args=(),
                        maxiter=maxiter, gtol=tol,
                        full_output=True)
        self.x, self.fval = stuff[0], stuff[1]
        self.time = time() - t0

    def argmin(self):
        return self.x

    def message(self):
        print('Scipy conjugate gradient implementation')
    

class ScipyNCG(object):

    def __init__(self, x, f, grad_f, hess_f, maxiter=None, tol=1e-7):
        t0 = time()
        stuff = fmin_ncg(f, x, grad_f, fhess=hess_f, args=(),
                         maxiter=maxiter, avextol=tol,
                         full_output=True)
        self.x, self.fval = stuff[0], stuff[1]
        self.time = time() - t0

    def argmin(self):
        return self.x

    def message(self):
        print('Scipy Newton conjugate gradient implementation')


class ScipyBFGS(object):

    def __init__(self, x, f, grad_f, maxiter=None, tol=1e-7):
        t0 = time()
        stuff = fmin_bfgs(f, x, fprime=grad_f, args=(),
                          maxiter=maxiter, gtol=tol,
                          full_output=True)
        self.x, self.fval = stuff[0], stuff[1]
        self.time = time() - t0

    def argmin(self):
        return self.x

    def message(self):
        print('Scipy BFGS quasi-Newton implementation')
    
    
