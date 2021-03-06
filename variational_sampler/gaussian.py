"""
A class to represent unnormalized Gaussian distributions.
"""

import numpy as np

from .numlib import (hdot, force_tiny, decomp_sym_matrix)


def Z_to_K(Z, dim, detV):
    return Z * force_tiny(detV) ** (-.5)\
        * (2 * np.pi) ** (-.5 * dim)


def K_to_Z(K, dim, detV):
    if detV < 0:
        return np.inf
    return K * (2 * np.pi) ** (.5 * dim) * detV ** .5


def _invV_to_theta(invV):
    A = -invV + .5 * np.diag(np.diagonal(invV))
    return A[np.triu_indices(A.shape[0])]


def _theta_to_invV(theta):
    dim = int(-1 + np.sqrt(1 + 8 * theta.size)) / 2
    A = np.zeros([dim, dim])
    A[np.triu_indices(dim)] = theta
    return -(A + A.T)


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

    def _set_dimensions(self, dim):
        """
        Set dimensional parameters
        """
        self._dim = dim
        self._theta_dim = (dim * (dim + 1)) / 2 + dim + 1

    def _set_moments(self, m, V, K=None, Z=None):
        m = np.asarray(m)
        dim = m.size
        self._set_dimensions(dim)

        # Mean and variance
        m = np.nan_to_num(np.reshape(m, (dim,)))
        V = np.nan_to_num(np.reshape(np.asarray(V), (dim, dim)))
        self._dim = dim
        self._m = m
        self._V = V

        # Compute the inverse and the square root of the variance
        # matrix
        abs_s, sign_s, P = decomp_sym_matrix(V)
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

    def _get_theta_dim(self):
        return self._theta_dim

    def _get_K(self):
        return self._K

    def _get_Z(self):
        """
        Compute the normalizing constant
        """
        return K_to_Z(self._K, self._dim, self._detV)

    def _get_m(self):
        return self._m

    def _get_V(self):
        return self._V

    def _get_invV(self):
        return self._invV

    def _get_sqrtV(self):
        return self._sqrtV

    def _get_theta(self):
        theta2 = _invV_to_theta(self._invV)
        theta1 = np.dot(self._invV, self._m)
        theta0 = np.log(self._K) - .5 * np.dot(self._m, theta1)
        return np.concatenate((np.array((theta0,)), theta1, theta2))

    def _set_theta(self, theta):
        """
        Convert theta to K, m, V
        """
        theta = np.asarray(theta)
        dim = _sample_dim(theta.size)
        self._set_dimensions(dim)
        invV = _theta_to_invV(theta[(dim + 1):])
        abs_s, sign_s, P = decomp_sym_matrix(invV)
        self._invV = invV
        inv_s = sign_s / abs_s
        self._V = np.dot(np.dot(P, np.diag(inv_s)), P.T)
        self._detV = np.prod(inv_s)
        self._sqrtV = np.dot(np.dot(P, np.diag(abs_s ** - .5)), P.T)
        self._m = np.dot(self._V, theta[1:(dim + 1)])
        self._K = np.exp(theta[0] + .5 * hdot(self._m, invV))

    def rescale(self, c):
        self._K *= c

    def mahalanobis(self, xs):
        if xs.ndim == 1:
            xs = np.reshape(xs, (1, xs.size))
        ys = (xs.T - self._m).T
        return np.sum(ys * np.dot(self._invV, ys), 0)

    def log(self, xs):
        return np.log(self._K) - .5 * self.mahalanobis(xs)

    def __call__(self, xs):
        """
        Evaluate the Gaussian at specified points.
        xs must have shape (dim, npts)
        """
        return self._K * np.exp(-.5 * self.mahalanobis(xs))

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(theta=self.theta + other.theta)
        elif hasattr(other, 'embed'):
            return self.__class__(theta=self.theta + other.embed().theta)
        else:
            raise ValueError('unsupported multiplitcation')

    def __div__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(theta=self.theta - other.theta)
        elif hasattr(other, 'embed'):
            return self.__class__(theta=self.theta - other.embed().theta)
        else:
            raise ValueError('unsupported division')

    def __pow__(self, power):
        return self.__class__(theta=power * self.theta)

    def sample(self, ndraws=1):
        """
        Return a d x N array.
        """
        xs = np.dot(self._sqrtV, np.random.normal(size=(self._dim, ndraws)))
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

    def integral(self):
        Z = self._get_Z()
        m = self._get_m()
        I1 = Z * m
        I2 = Z * (self._get_V()
                  + np.dot(m.reshape((self._dim, 1)),
                           m.reshape((1, self._dim))))[\
            np.triu_indices(self._dim)]
        return np.concatenate((np.array((Z,)), I1, I2))

    def __str__(self):
        s = 'Gaussian distribution with parameters:\n'
        s += str(self._get_Z()) + '\n'
        s += str(self._m) + '\n'
        s += str(self._V) + '\n'
        return s

    def copy(self):
        return self.__class__(self._m, self._V, K=self._K)

    dim = property(_get_dim)
    theta_dim = property(_get_theta_dim)
    K = property(_get_K)
    Z = property(_get_Z)
    m = property(_get_m)
    V = property(_get_V)
    invV = property(_get_invV)
    sqrtV = property(_get_sqrtV)
    theta = property(_get_theta, _set_theta)


class FactorGaussian(object):

    def __init__(self, m=None, v=None, K=None, Z=None, theta=None):
        if not theta == None:
            self._set_theta(theta)
        else:
            self._set_moments(m, v, K, Z)

    def _set_dimensions(self, dim):
        self._dim = dim
        self._theta_dim = 2 * dim + 1

    def _set_moments(self, m, v, K=None, Z=None):
        m = np.asarray(m)
        dim = m.size
        self._set_dimensions(dim)

        # Mean and variance
        m = np.nan_to_num(np.reshape(m, (dim,)))
        v = np.nan_to_num(np.reshape(v, (dim,)))
        self._dim = dim
        self._m = m
        self._v = v
        self._invv = np.nan_to_num(1 / self._v)
        self._detV = np.prod(v)

        # Normalization constant
        if not K == None:
            self._K = float(K)
        else:
            if Z == None:
                Z = 1.0
            self._K = Z_to_K(Z, self._dim, self._detV)

    def _get_dim(self):
        return self._dim

    def _get_theta_dim(self):
        return self._theta_dim

    def _get_K(self):
        return self._K

    def _get_Z(self):
        return K_to_Z(self._K, self._dim, self._detV)

    def _get_m(self):
        return self._m

    def _get_V(self):
        return np.diag(self._v)

    def _get_v(self):
        return self._v

    def _get_invV(self):
        return np.diag(self._invv)

    def _get_sqrtV(self):
        return np.diag(np.sqrt(np.abs(self._v)))

    def _get_theta(self):
        invV = np.nan_to_num(1 / self._v)
        theta2 = -.5 * invV
        theta1 = invV * self._m
        theta0 = np.log(self._K) - .5 * np.dot(self._m, theta1)
        return np.concatenate((np.array((theta0,)), theta1, theta2))

    def _set_theta(self, theta):
        theta = np.asarray(theta)
        dim = (theta.size - 1) / 2
        self._set_dimensions(dim)
        invv = -2 * theta[(dim + 1):]
        self._invv = invv
        self._v = np.nan_to_num(1 / invv)
        self._m = self._v * theta[1:(dim + 1)]
        self._K = np.exp(theta[0] + .5 * np.dot(self._m, invv * self._m))
        self._detV = np.prod(self._v)

    def rescale(self, c):
        self._K *= c

    def mahalanobis(self, xs):
        if xs.ndim == 1:
            xs = np.reshape(xs, (1, xs.size))
        ys = (xs.T - self._m).T
        return np.sum(self._invv * (ys ** 2).T, 1)

    def log(self, xs):
        return np.log(self._K) - .5 * self.mahalanobis(xs)

    def __call__(self, xs, log=True):
        """
        Evaluate the Gaussian at specified points.
        xs must have shape (dim, npts)
        """
        return self._K * np.exp(-.5 * self.mahalanobis(xs))

    def sample(self, ndraws=1):
        xs = (np.sqrt(np.abs(self._v)) * \
                  np.random.normal(size=(self._dim, ndraws)).T).T
        return (self._m + xs.T).T  # preserves shape

    def __str__(self):
        s = 'Factored Gaussian distribution with parameters:\n'
        s += str(self._get_Z()) + '\n'
        s += str(self._m) + '\n'
        s += 'diag(' + str(self._v) + ')\n'
        return s

    def embed(self):
        """
        Return equivalent instance of the parent class
        """
        return Gaussian(self.m, self.V, K=self.K)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(theta=self.theta + other.theta)
        elif isinstance(other, Gaussian):
            return Gaussian(theta=self.embed().theta + other.theta)
        else:
            raise ValueError('unsupported multiplication')

    def __div__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(theta=self.theta - other.theta)
        elif isinstance(other, Gaussian):
            return Gaussian(theta=self.embed().theta - other.theta)
        else:
            raise ValueError('unsupported division')

    def __pow__(self, power):
        return self.__class__(theta=power * self.theta)

    def kl_div(self, other):
        return self.embed().kl_div(other)

    def integral(self):
        Z = self._get_Z()
        m = self._get_m()
        I1 = Z * m
        I2 = Z * (self._get_v() + m ** 2)
        return np.concatenate((np.array((Z,)), I1, I2))

    def copy(self):
        return self.__class__(self._m, self._v, K=self._K)

    dim = property(_get_dim)
    theta_dim = property(_get_theta_dim)
    K = property(_get_K)
    Z = property(_get_Z)
    m = property(_get_m)
    V = property(_get_V)
    v = property(_get_v)
    invV = property(_get_invV)
    sqrtV = property(_get_sqrtV)
    theta = property(_get_theta, _set_theta)


class GaussianFamily(object):

    def __init__(self, dim):
        self.dim = dim
        self.theta_dim = (dim * (dim + 1)) / 2 + dim + 1

    def design_matrix(self, pts):
        """
        pts: array with shape (dim, n)
        """
        I, J = np.triu_indices(pts.shape[0])
        F = np.array([pts[i, :] * pts[j, :] for i, j in zip(I, J)])
        return np.concatenate((np.ones((1, pts.shape[1])), pts, F))

    def from_integral(self, integral):
        Z = integral[0]
        m = integral[1: (self.dim + 1)] / Z
        V = np.zeros((self.dim, self.dim))
        idx = np.triu_indices(self.dim)
        V[idx] = integral[(self.dim + 1):] / Z
        V.T[np.triu_indices(self.dim)] = V[idx]
        V -= np.dot(m.reshape(m.size, 1), m.reshape(1, m.size))
        return Gaussian(m, V, Z=Z)

    def from_theta(self, theta):
        return Gaussian(theta=theta)

    def check(self, obj):
        return isinstance(obj, Gaussian)


class FactorGaussianFamily(object):

    def __init__(self, dim):
        self.dim = dim
        self.theta_dim = 2 * dim + 1

    def design_matrix(self, pts):
        """
        pts: array with shape (dim, n)
        """
        return np.concatenate((np.ones((1, pts.shape[1])), pts,  pts ** 2))

    def from_integral(self, integral):
        Z = integral[0]
        m = integral[1: (self.dim + 1)] / Z
        v = integral[(self.dim + 1):] / Z - m ** 2
        return FactorGaussian(m, v, Z=Z)

    def from_theta(self, theta):
        return FactorGaussian(theta=theta)

    def check(self, obj):
        return isinstance(obj, FactorGaussian)
