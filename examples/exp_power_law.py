import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler.variational_sampler import (VariationalFit,
                                                     StraightFit)
from variational_sampler.gaussian_process import GaussianProcessFit
from variational_sampler.gaussian import Gaussian
from variational_sampler.sampling import Sample
from variational_sampler.toy_examples import ExponentialPowerLaw

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

def display_fits(sample, target, methods, colors, acronyms, local=False):
    xs = sample.x
    xmax = int(np.max(np.abs(xs.squeeze()))) + 1
    xmin = -xmax
    x = np.linspace(-xmax, xmax, 2 * xmax / 0.01)
    x = np.reshape(x, (1, x.size))
    plt.figure()
    if local:
        fits = [m.loc_fit(x) for m in methods]
    else:
        fits = [m.fit(x) for m in methods]
    for fit, color in zip(fits, colors):
        plt.plot(x.squeeze(), fit, color, linewidth=2)
    plt.legend(acronyms)
    target_xs = target(xs.squeeze())
    target_x = target(x.squeeze())
    if local:
        target_xs *= sample.kernel(xs)
        target_x *= sample.kernel(x)
    plt.stem(xs.squeeze(), target_xs, linefmt='k-', markerfmt='ko')
    plt.plot(x.squeeze(), target_x, 'k')
    plt.show()


target = ExponentialPowerLaw(beta=BETA)
v = target.V.squeeze()
h2 = 10 * v

s = Sample((0, h2), ndraws=NPTS)
vf = VariationalFit(target, s, maxiter=10)
sf = StraightFit(target, s)
gf = GaussianProcessFit(target, s, var=v)

gs_loc_fit = gauss_hermite(target, h2, 250)
gh_loc_fit = gauss_hermite(target, h2, NPTS)

print('Error for VS: %f (expected: %f)'\
          % (gs_loc_fit.kl_div(vf.loc_fit), vf.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gs_loc_fit.kl_div(sf.loc_fit), sf.kl_error))
print('Error for BMC: %f'% gs_loc_fit.kl_div(gf.loc_fit))
print('Error for GH: %f' % gs_loc_fit.kl_div(gh_loc_fit))


acronyms = ('VS', 'IS', 'BMC')
colors = ('blue', 'orange', 'green')
display_fits(s, target, (vf, sf, gf), colors, acronyms)
display_fits(s, target, (vf, sf, gf), colors, acronyms, local=True)
