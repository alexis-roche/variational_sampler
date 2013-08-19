import numpy as np

from .numlib import force_tiny, inv_sym_matrix


class GaussianMixture(object):

    def __init__(self, weights, gaussians):
        """
        centers: array of shape (d, N)
        weights: array of shape (N,)
        V: componentwise variance matrix, array of shape (d,d)
        """
        # Dimension
        dim = gaussians[0].m.size
        self._dim = dim
        self._weights = np.asarray(weights)
        self._gaussians = gaussians
        self._moments()

    def _moments(self):
        Z = 0.0
        m = np.zeros(self._dim)
        V = np.zeros((self._dim, self._dim))
        for w, g in zip(self._weights, self._gaussians):
            Z += w * g.Z
            m += w * g.Z * g.m
            V += w * g.Z * (g.V + np.dot(g.m.reshape((len(g.m), 1)),
                                         g.m.reshape((1, len(g.m)))))
        Z_safe = force_tiny(Z)
        m /= Z_safe
        V = V / Z_safe - np.dot(m.reshape((len(m), 1)), m.reshape((1, len(m))))
        self._Z = Z
        self._m = m
        self._V = V

    def __call__(self, xs):
        """
        Sample q(x) at specified points.
        xs must be two-dimensional with shape[0] equal to self.dim.
        """
        ret = np.zeros(xs.shape[1])
        for w, g in zip(self._weights, self._gaussians):
            ret += w * g(xs)
        return ret

    def _get_dim(self):
        return self._dim

    def _get_Z(self):
        return self._Z

    def _get_m(self):
        return self._m

    def _get_V(self):
        return self._V

    def _get_invV(self):
        return inv_sym_matrix(self._V)

    dim = property(_get_dim)
    m = property(_get_m)
    V = property(_get_V)
    invV = property(_get_invV)
    Z = property(_get_Z)
