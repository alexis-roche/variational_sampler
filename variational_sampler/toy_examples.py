import numpy as np
from scipy.special import gamma


def norm(x, m, beta=2.):
    """
    x is an array with shape (d, N)
    m is an array with shape (d,)
    Returns (x-m)**2 as an array of shape (N,)
    """
    if x.ndim == 1:
        return np.abs(x) ** beta
    tmp = np.abs((x.T - m).T)  # (d, N)
    return np.sum(tmp ** beta, 0)


class ExponentialPowerLaw():

    def __init__(self, alpha=np.sqrt(2), beta=2.0, dim=1):

        self.dim = int(dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # ideal Gaussian parameters
        self.Z = 1.
        self.m = np.zeros((self.dim, ))
        self.v = (self.alpha ** 2) * \
            gamma(3 / self.beta) / gamma(1 / self.beta)
        self.V = self.v * np.eye(self.dim)
        self.K = (self.beta / (2 * self.alpha * gamma(1 / self.beta))) \
                      ** self.dim

    def __call__(self, x):
        return self.K * (np.exp(-norm(x / self.alpha, 0, beta=self.beta)))

    def draw(self):
        return np.sqrt(self.v) * np.squeeze(np.random.normal(size=self.dim))
