import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler.variational_sampler import VariationalFit
from variational_sampler.importance_sampler import ImportanceFit
from variational_sampler.gp_sampler import GaussianProcessFit
from variational_sampler.gaussian import Gaussian
from variational_sampler.sampling import Sample
from _toy_dist import ExponentialPowerLaw
from _display import display_fit

BETA = .5
NPTS = 10


def gauss_hermite(target, h2, npts):
    x, w = h_roots(npts)
    x = x.real * np.sqrt(2 * h2)
    c = 1 / np.sqrt(np.pi)
    Z = c * np.sum(w * target(x))
    m = c * np.sum(w * x * target(x)) / Z
    V = c * np.sum(w * (x ** 2) * target(x)) / Z - m ** 2
    return Gaussian(m, V, Z=Z)

target = ExponentialPowerLaw(beta=BETA)
v = target.V.squeeze()
h2 = 10 * v

s = Sample((0, h2), context='kernel', ndraws=NPTS)
vs = VariationalFit(target, s, maxiter=10)
v0 = ImportanceFit(target, s)
v1 = GaussianProcessFit(target, s, var=v)

gs_loc_fit = gauss_hermite(target, h2, 250)
gh_loc_fit = gauss_hermite(target, h2, NPTS)

print('Error for VS: %f (expected: %f)'\
          % (gs_loc_fit.kl_div(vs.loc_fit), vs.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gs_loc_fit.kl_div(v0.loc_fit), v0.kl_error))
print('Error for BMC: %f'% gs_loc_fit.kl_div(v1.loc_fit))
print('Error for GH: %f' % gs_loc_fit.kl_div(gh_loc_fit))


acronyms = ('VS', 'IS', 'BMC')
colors = ('blue', 'orange', 'green')
plt.figure()
display_fit(s, target, (vs, v0, v1), colors, acronyms)
plt.title('global fits')
plt.figure()
display_fit(s, target, (vs, v0, v1), colors, acronyms, local=True)
plt.title('local fits')
plt.show()
