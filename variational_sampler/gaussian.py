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
        Set dimensional parameters: np (number of parameters), dim2
        (number of second-order parameters)
        """
        self._dim = dim
        self._dim2 = (dim * (dim + 1)) / 2
        self._theta_dim = self._dim2 + dim + 1

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
        return np.concatenate((theta2, theta1, np.array((theta0,))))

    def _set_theta(self, theta):
        """
        Convert new theta back into K, m, V
        """
        theta = np.asarray(theta)
        dim = _sample_dim(theta.size)
        self._set_dimensions(dim)
        invV = _theta_to_invV(theta[0:self._dim2])
        abs_s, sign_s, P = decomp_sym_matrix(invV)
        self._invV = invV
        inv_s = sign_s / abs_s
        self._V = np.dot(np.dot(P, np.diag(inv_s)), P.T)
        self._detV = np.prod(inv_s)
        self._sqrtV = np.dot(np.dot(P, np.diag(abs_s ** - .5)), P.T)
        self._m = np.dot(self._V, theta[self._dim2:-1])
        self._K = np.exp(theta[-1] + .5 * hdot(self._m, invV))

    def __call__(self, xs):
        """
        Sample q(x) at specified points.
        xs must be two-dimensional with shape[0] equal to self.dim.
        """
        if xs.ndim == 1:
            xs = np.reshape(xs, (1, xs.size))
        ys = (xs.T - self._m).T
        u2 = np.sum(ys * np.dot(self._invV, ys), 0)
        return self._K * np.exp(-.5 * u2)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(theta=self.theta + other.theta)
        elif hasattr(other, 'embed'):
            return self.__class__(theta=self.theta + other.embed().theta)
        else:
            raise ValueError('wrong multiplication argument')

    def __div__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(theta=self.theta - other.theta)
        elif hasattr(other, 'embed'):
            return self.__class__(theta=self.theta - other.embed().theta)
        else:
            raise ValueError('wrong division argument')
        
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

    def __str__(self):
        s = 'Gaussian distribution with parameters:\n'
        s += str(self._get_Z()) + '\n'
        s += str(self._m) + '\n'
        s += str(self._V) + '\n'
        return s

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

    def __init__(self, m=None, V=None, K=None, Z=None, theta=None):
        if not theta == None:
            self._set_theta(theta)
        else:
            self._set_moments(m, V, K, Z)

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
        return np.concatenate((theta2, theta1, np.array((theta0,))))

    def _set_theta(self, theta):
        theta = np.asarray(theta)
        dim = (theta.size - 1) / 2
        self._set_dimensions(dim)
        invv = -2 * theta[0:self._dim]
        self._invv = invv
        self._v = np.nan_to_num(1 / invv)
        self._m = self._v * theta[self._dim:-1]
        self._K = np.exp(theta[-1] + .5 * np.dot(self._m, invv * self._m))
        self._detV = np.prod(self._v)

    def __call__(self, xs):
        if xs.ndim == 1:
            xs = np.reshape(xs, (1, xs.size))
        ys = (xs.T - self._m).T
        u2 = np.sum(self._invv * (ys ** 2).T, 1)
        return self._K * np.exp(-.5 * u2)

    def sample(self, ndraws=1):
        xs = (np.sqrt(np.abs(self._v)) * np.random.normal(size=(self._dim, ndraws)).T).T
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
            raise ValueError('wrong multiplication argument')

    def __div__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(theta=self.theta - other.theta)
        elif instance(other, Gaussian):
            return Gaussian(theta=self.embed().theta - other.theta)
        else:
            raise ValueError('wrong division argument')

    def __pow__(self, power):
        return self.__class__(theta=power * self.theta)

    def kl_div(self, other):
        return self.embed().kl_div(other)


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
        self._dim = dim

    def design_matrix(self, pts):
        """
        pts: array with shape (dim, n)
        """
        I, J = np.triu_indices(pts.shape[0])
        F = np.array([pts[i, :] * pts[j, :] for i, j in zip(I, J)])
        return np.concatenate((F, pts, np.ones((1, pts.shape[1]))))

    def from_moment(self, moment):
        Z = moment[-1]
        m = moment[-1 - self._dim:-1] / Z
        V = np.zeros((self._dim, self._dim))
        idx = np.triu_indices(self._dim) 
        V[idx] = moment[0:-1 - self._dim] / Z
        V[np.tril_indices(self._dim)] = V[idx]
        V -= np.diag(m ** 2)
        return Gaussian(m, V, Z=Z)
        
    def from_theta(self, theta):
        return Gaussian(theta=theta)

    def check(self, obj):
        return isinstance(obj, Gaussian)


class FactorGaussianFamily(object):

    def __init__(self, dim):
        self._dim = dim

    def design_matrix(self, pts):
        """
        pts: array with shape (dim, n)
        """
        return np.concatenate((pts ** 2,  pts, np.ones((1, pts.shape[1]))))

    def from_moment(self, moment):
        Z = moment[-1]
        m = moment[-1 - self._dim:-1] / Z
        v = moment[0:-1 - self._dim] / Z - m ** 2
        return FactorGaussian(m, v, Z=Z)
        
    def from_theta(self, theta):
        return FactorGaussian(theta=theta)

    def check(self, obj):
        return isinstance(obj, FactorGaussian)
