"""
A class to represent unnormalized Gaussian distributions.
"""

import numpy as np

from .numlib import (hdot, force_tiny, safe_eigh)


def Z_to_K(Z, dim, detV):
    return Z * force_tiny(detV) ** (-.5)\
        * (2 * np.pi) ** (-.5 * dim)


def K_to_Z(K, dim, detV):
    if detV < 0:
        return np.inf
    return K * (2 * np.pi) ** (.5 * dim) * detV ** .5


def invV_to_theta(invV):
    A = -invV + .5 * np.diag(np.diagonal(invV))
    return A[np.triu_indices(A.shape[0])]


def theta_to_invV(theta):
    dim = int(-1 + np.sqrt(1 + 8 * theta.size)) / 2
    A = np.zeros([dim, dim])
    A[np.triu_indices(dim)] = theta
    return -(A + A.T)


def expand_parameter(theta, P):
    """
    g(x) = g0(P x)

    P: n x N

    theta' P x = (P' theta)' x
    (Px)' invV Px = x' P' invV P x
    """
    if P == None:
        return theta
    elif P.ndim < 2:
        P = np.reshape(P, (1, P.size))
    dim = P.shape[0]
    dim2 = (dim * (dim + 1)) / 2

    theta0 = theta[-1]
    theta1 = np.dot(P.T, theta[dim2:-1])
    aux = theta_to_invV(theta[0:dim2])
    invV = -.5 * np.dot(np.dot(P.T, aux), P)
    theta2 = invV_to_theta(invV)
    print theta2

    return np.concatenate((theta2, theta1, np.array((theta0,))))


def _param_dim(d):
    return 1 + d + (d * (d + 1)) / 2


def _sample_dim(dim):
    return int(-1.5 + np.sqrt(.25 + 2 * dim))


class Gaussian(object):
    """
    A class to describe unnormalized Gaussian distributions under the
    form:

    g(x) = K exp[(x-m)'*A*(x-m)] with A = -.5*inv(V)
    """

    def __init__(self, m=None, V=None, K=None, Z=None, theta=None):

        # If theta is provided, ignore other parameters
        if not theta == None:
            self._set_theta(theta)
        else:
            self._set_moments(m, V, K, Z)

    def _set_dimension_parameters(self, dim):
        """
        Set dimensional parameters: np (number of parameters), dim2
        (number of second-order parameters)
        """
        self._dim = dim
        self._dim2 = (dim * (dim + 1)) / 2
        self._np = self._dim2 + dim + 1
        # Useful arrays for indexing
        # I, J = np.mgrid[0:dim, 0:dim]
        # self._Iu, self._Ju = np.where((I - J) <= 0)

    def _set_moments(self, m, V, K=None, Z=None):
        m = np.asarray(m)
        dim = m.size
        self._set_dimension_parameters(dim)

        # Mean and variance
        m = np.reshape(m, (dim,))
        V = np.reshape(np.asarray(V), (dim, dim))
        self._dim = dim
        self._m = np.nan_to_num(m)
        self._V = np.nan_to_num(V)

        # Compute the inverse and the square root of the variance
        # matrix
        abs_s, sign_s, P = safe_eigh(V)
        self._invV = np.dot(np.dot(P, np.diag(sign_s / abs_s)), P.T)
        self._detV = np.prod(abs_s * sign_s)
        self._sqrtV = np.dot(np.dot(P, np.diag(abs_s ** .5)), P.T)

        # Normalization constant
        if not K == None:
            self._K = float(K)
        else:
            if Z == None:
                Z = 1.0
            self._K = Z_to_K(Z, self._dim, self._detV)

    def _get_dim(self):
        return self._dim

    def _get_param_dim(self):
        return _param_dim(self._dim)

    def _get_m(self):
        return self._m

    def _get_V(self):
        return self._V

    def _get_invV(self):
        return self._invV

    def _get_sqrtV(self):
        return self._sqrtV

    def _get_K(self):
        return self._K

    def _get_theta(self):
        theta2 = invV_to_theta(self._invV)
        theta1 = np.dot(self._invV, self._m)
        theta0 = np.log(self._K) - .5 * hdot(self._m, self._invV)
        return np.concatenate((theta2, theta1, np.array((theta0,))))

    def _set_theta(self, theta):
        """
        Convert new theta back into K, m, V
        """
        theta = np.asarray(theta)
        dim = _sample_dim(theta.size)
        self._set_dimension_parameters(dim)
        invV = theta_to_invV(theta[0:self._dim2])
        abs_s, sign_s, P = safe_eigh(invV)
        self._invV = invV
        inv_s = sign_s / abs_s
        self._V = np.dot(np.dot(P, np.diag(inv_s)), P.T)
        self._detV = np.prod(inv_s)
        self._sqrtV = np.dot(np.dot(P, np.diag(abs_s ** - .5)), P.T)
        self._m = np.dot(self._V, theta[self._dim2:-1])
        self._K = np.exp(theta[-1] + .5 * hdot(self._m, invV))

    def _get_Z(self):
        """
        Compute \int q(x) dx over R^n where log q(x) is the quadratic
        fit.  Returns np.inf if the variance is not positive definite.
        """
        return K_to_Z(self._K, self._dim, self._detV)

    def __call__(self, xs):
        """
        Sample q(x) at specified points.
        xs must be two-dimensional with shape[0] equal to self.dim.
        """
        ys = (xs.T - self._m).T
        u2 = np.sum(ys * np.dot(self._invV, ys), 0)
        return force_tiny(self._K * np.exp(-.5 * u2))

    def copy(self):
        return Gaussian(self._m, self._V, self._K)

    def __mul__(self, other):
        ret = self.copy()
        ret._set_theta(self.theta + other.theta)
        return ret

    def __pow__(self, power):
        ret = self.copy()
        ret._set_theta(power * self.theta)
        return ret

    def sample(self, ndraws=1):
        """
        Return a d x N array.
        """
        xs = np.dot(self._sqrtV, np.random.normal(size=[self._dim, ndraws]))
        return (self._m + xs.T).T  # preserves shape

    def kl_div(self, other):
        """
        Return the kl divergence D(self, other) where other is another
        Gaussian instance.
        """
        dm = self.m - other.m
        dV = np.dot(other.invV, self.V)
        err = -np.log(force_tiny(np.linalg.det(dV)))
        err += np.sum(np.diag(dV)) - dm.size
        err += np.dot(dm.T, np.dot(other.invV, dm))
        err = np.maximum(.5 * err, 0.0)
        if np.isinf(other.Z):
            return np.inf
        z_err = np.maximum(self.Z * np.log(self.Z / force_tiny(other.Z))
                           + other.Z - self.Z, 0.0)
        return self.Z * err + z_err

    def __str__(self):
        s = 'Gaussian distribution with parameters:\n'
        s += str(self.Z) + '\n'
        s += str(self.m) + '\n'
        s += str(self.V) + '\n'
        return s

    dim = property(_get_dim)
    param_dim = property(_get_param_dim)
    K = property(_get_K)
    m = property(_get_m)
    V = property(_get_V)
    Z = property(_get_Z)
    invV = property(_get_invV)
    sqrtV = property(_get_sqrtV)
    theta = property(_get_theta, _set_theta)
