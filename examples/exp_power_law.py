import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian
from _toy_dist import ExponentialPowerLaw
from _display import display_fit

BETA = 3
NPTS = 10

def gauss_hermite(target, h2, npts):
    p = lambda x: np.exp(target(x))
    x, w = h_roots(npts)
    x = x.real * np.sqrt(2 * h2)
    c = 1 / np.sqrt(np.pi)
    Z = c * np.sum(w * p(x))
    m = c * np.sum(w * x * p(x)) / Z
    V = c * np.sum(w * (x ** 2) * p(x)) / Z - m ** 2
    return Gaussian(m, V, Z=Z)

target = ExponentialPowerLaw(beta=BETA)
v = target.V
h2 = 10 * v

vs = VariationalSampler(target, (0, h2), NPTS, context='kernel')
f = vs.fit()
f0 = vs.fit('l')
f2 = vs.fit('gp', var=v)

gs_glob_fit = gauss_hermite(target, h2, 250)
gh_glob_fit = gauss_hermite(target, h2, NPTS)

print('Error for VS: %f (expected: %f)'\
          % (gs_glob_fit.kl_div(f.glob_fit), f.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gs_glob_fit.kl_div(f0.glob_fit), f0.kl_error))
print('Error for BMC: %f'% gs_glob_fit.kl_div(f2.glob_fit))
print('Error for GH: %f' % gs_glob_fit.kl_div(gh_glob_fit))


acronyms = ('VS', 'IS', 'BMC')
colors = ('blue', 'orange', 'green')
plt.figure()
display_fit(vs, target, (f, f0, f2), colors, acronyms)
plt.title('global fits')
plt.figure()
display_fit(vs, target, (f, f0, f2), colors, acronyms, local=True)
plt.title('local fits')
plt.show()
