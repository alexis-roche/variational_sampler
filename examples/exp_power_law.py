import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler.variational_sampler import (VariationalFit,
                                                     ImportanceFit)
from variational_sampler.gaussian import Gaussian
from variational_sampler.sampling import Sample
from variational_sampler.toy_examples import ExponentialPowerLaw



def gauss_hermite(target, h2, npts):
    x, w = h_roots(npts)
    x = x.real * np.sqrt(2 * h2)
    c = 1 / np.sqrt(np.pi)
    Z = c * np.sum(w * target(x))
    m = c * np.sum(w * x * target(x)) / Z
    V = c * np.sum(w * (x ** 2) * target(x)) / Z - m ** 2
    return Gaussian(m, V, Z=Z)


BETA = 2
NPTS = 100

target = ExponentialPowerLaw(beta=BETA)
h2 = 10 * target.V.squeeze()

s = Sample(target, 0, h2, ndraws=NPTS)
vs = VariationalFit(s, maxiter=10)
ds = ImportanceFit(s)

gs_loc_fit = gauss_hermite(target, h2, 250)
gh_loc_fit = gauss_hermite(target, h2, NPTS)

print ('Error for VS: %f (expected: %f)'\
           % (gs_loc_fit.kl_div(vs.loc_fit), vs.kl_error))
print ('Error for IS: %f (expected: %f)'\
           % (gs_loc_fit.kl_div(ds.loc_fit), ds.kl_error))
print ('Error for GH: %f' % gs_loc_fit.kl_div(gh_loc_fit))

xs = vs.sample.x
xmax = int(np.max(np.abs(xs.squeeze()))) + 1
xmin = -xmax
x = np.linspace(-xmax, xmax, 2 * xmax / 0.01)
x = np.reshape(x, (1, x.size))
plt.plot(x.squeeze(), vs.fit(x), 'blue', linewidth=2)
plt.plot(x.squeeze(), ds.fit(x), 'orange', linewidth=2)
plt.legend(('VS', 'IS'))
plt.stem(xs.squeeze(), target(xs.squeeze()), linefmt='k-', markerfmt='ko')
plt.plot(x.squeeze(), target(x.squeeze()), 'k')
