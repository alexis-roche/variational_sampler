"""
Constants used in several modules.
Basic implementation of Gauss-Newton gradient descent scheme.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import fmin_cg, fmin_ncg, fmin_bfgs

XTOL = 1e-7
GTOL = 1e-5
dinfo = np.finfo(np.double)
TINY = dinfo.tiny
HUGE = dinfo.max


def force_tiny(x):
    return np.maximum(x, TINY)


def safe_eigh(A):
    s, P = eigh(A)
    sign_s = 2. * (s >= 0) - 1
    abs_s = force_tiny(np.abs(s))
    return abs_s, sign_s, P


def newton(f, x0, fprime=None, hess=None, args=(),
           maxiter=None, xtol=XTOL, callback=None):

    """
    Custom Newton method with adaptive step size.
    """
    x = np.asarray(x0).flatten()
    fval0 = f(x, *args)
    fval = fval0
    it = 0
    if maxiter == None:
        maxiter = np.inf

    while it < maxiter:

        # Update iteration number
        it += 1

        # Compute gradient and hessian at current point
        xN = x
        fvalN = fval
        g = fprime(xN, *args)
        H = hess(xN, *args)

        # Solve H*dx = -g
        try:  # Use Cholesky decomposition: H = L L.T
            L, _ = cho_factor(H, lower=0)
            dx = - cho_solve((L, 0), g)
        except:  # Assume diagonal H = tr(H)/n Id
            print('Ooops... singular Hessian, regularizing')
            trH = np.nan_to_num(force_tiny(np.trace(H)))
            dx = -(H.shape[0] / trH) * g
        dx = np.nan_to_num(dx)

        # Adaptive Newton proposal
        unbeaten = True
        sub_it = 0
        while unbeaten:
            sub_it += 1
            # stop if step size is too small
            step_size = np.max(np.abs(dx))
            if step_size < xtol:
                break
            # new proposal
            x = xN + dx
            fval = f(x, *args)
            unbeaten = (fval > fvalN)
            # reduce step size for next try
            dx *= .5
        if unbeaten:
            x = xN
            break
        x = np.nan_to_num(x)
        step_size = np.max(np.abs(dx))
        if step_size < xtol:
            break

    print('Number of iterations: %d' % it)
    print('Gradient norm: %f' % np.max(np.abs(g)))
    print('Minimum criterion value: %f' % fval)

    return x


def minimize(f, x, grad,
             hess=None,
             args=(),
             minimizer='newton',
             maxiter=None,
             xtol=XTOL,
             gtol=GTOL):
    """
    Util function to call any minimizer with the desired arguments
    """
    if minimizer == 'newton':
        xm = newton(f, x, grad, hess=hess, args=args,
                      maxiter=maxiter, xtol=xtol)
    elif minimizer == 'ncg':
        xm = fmin_ncg(f, x, grad, fhess=hess, args=args,
                        maxiter=maxiter, avextol=xtol)
    elif minimizer == 'cg':
        xm = fmin_cg(f, x, fprime=grad, args=args,
                       maxiter=maxiter, gtol=gtol)
    elif minimizer == 'bfgs':
        xm = fmin_bfgs(f, x, fprime=grad, args=args,
                         maxiter=maxiter, gtol=gtol)
    else:
        raise ValueError('unknown minimizer')
    return np.nan_to_num(xm)
