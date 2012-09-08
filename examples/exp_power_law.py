import numpy as np
from scipy.special import gamma
import pylab as plt
from variational_sampler import VariationalSampler

BETA = 3
NDRAWS = 10


def norm(x, m, beta=2.):
    """
    x is an array with shape (d, N)
    m is an array with shape (d,)
    Returns (x-m)**2 as an array of shape (N,)
    """
    tmp = np.abs((x.T - m).T)  # (d, N)
    return np.sum(tmp ** beta, 0)


class ExponentialPowerLaw():

    def __init__(self, alpha=np.sqrt(2), beta=2.0, d=1):

        self.d = int(d)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # ideal Gaussian parameters
        self.Z = 1.
        self.m = np.zeros((self.d, ))
        self.v = (self.alpha ** 2) * \
            gamma(3 / self.beta) / gamma(1 / self.beta)
        self.V = self.v * np.eye(self.d)
        self.K = (self.beta / (2 * self.alpha * gamma(1 / self.beta))) \
                      ** self.d

    def dist(self, x):
        return self.K * (np.exp(-norm(x / self.alpha, 0, beta=self.beta)))

    def draw(self):
        return np.sqrt(self.v) * np.squeeze(np.random.normal(size=self.d))




target = ExponentialPowerLaw(beta=BETA)
vs = VariationalSampler(target.dist, 0, 10 * target.V,
                        ndraws=NDRAWS)
fit = vs.get_fit()

xs = vs.sample.x
xmax = int(np.max(np.abs(xs.squeeze()))) + 1
xmin = -xmax
x = np.linspace(-xmax, xmax, 2 * xmax / 0.01)
x = np.reshape(x, (1, x.size))
plt.stem(xs.squeeze(), target.dist(xs), linefmt='k-', markerfmt='ko')
plt.plot(x.squeeze(), fit(x), 'orange', linewidth=2)
plt.plot(x.squeeze(), target.dist(x), 'k')
