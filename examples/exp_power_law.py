import numpy as np
import pylab as plt
from variational_sampler import (VariationalFit,
                                 DirectFit,
                                 Gaussian,
                                 Sample)
from scipy.special.orthogonal import h_roots
from scipy.special import gamma


class ExponentialPowerLaw(object):
    def __init__(self, alpha=np.sqrt(2), beta=2, Z=1):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.Z = float(Z)
        self.m = 0
        self.V = (self.alpha ** 2) * \
            gamma(3 / self.beta) / gamma(1 / self.beta)
        self.K = self.Z * (self.beta / (2 * self.alpha * gamma(1 / self.beta)))
        
    def __call__(self, x):
        return self.K * np.exp(-np.abs(x / self.alpha) ** self.beta)


def gauss_hermite(target, h2, npts):
    x, w = h_roots(npts)
    x = x.real * np.sqrt(2 * h2)
    c = 1 / np.sqrt(np.pi)
    Z = c * np.sum(w * target(x))
    m = c * np.sum(w * x * target(x)) / Z
    V = c * np.sum(w * (x ** 2) * target(x)) / Z - m ** 2
    return Gaussian(m, V, Z=Z)


BETA = 1.2
NPTS = 20

target = ExponentialPowerLaw(beta=BETA)
h2 = 10 * target.V.squeeze()

s = Sample(target, 0, h2, ndraws=NPTS)
vs = VariationalFit(s)
ds = DirectFit(s)

gs_loc_fit = gauss_hermite(target, h2, 250)
gh_loc_fit = gauss_hermite(target, h2, NPTS)

for m, f in zip(('VS', 'DS', 'GH'),
                (vs.loc_fit, ds.loc_fit, gh_loc_fit)):
     err = gs_loc_fit.kl_div(f)
     print ('Error for %s: %f' % (m, err))

xs = vs.sample.x
xmax = int(np.max(np.abs(xs.squeeze()))) + 1
xmin = -xmax
x = np.linspace(-xmax, xmax, 2 * xmax / 0.01)
x = np.reshape(x, (1, x.size))
plt.plot(x.squeeze(), vs.fit(x), 'blue', linewidth=2)
plt.plot(x.squeeze(), ds.fit(x), 'orange', linewidth=2)
plt.legend(('VS', 'DS'))
plt.stem(xs.squeeze(), target(xs.squeeze()), linefmt='k-', markerfmt='ko')
plt.plot(x.squeeze(), target(x.squeeze()), 'k')
