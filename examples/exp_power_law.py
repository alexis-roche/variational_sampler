import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler.variational_sampler import (VariationalFit,
                                                     VariationalFitIS)
from variational_sampler.bayesian_monte_carlo import VariationalFitBMC
from variational_sampler.gaussian import Gaussian
from variational_sampler.sampling import Sample
from variational_sampler.toy_examples import ExponentialPowerLaw

BETA = 1
NPTS = 100

def gauss_hermite(target, h2, npts):
    x, w = h_roots(npts)
    x = x.real * np.sqrt(2 * h2)
    c = 1 / np.sqrt(np.pi)
    Z = c * np.sum(w * target(x))
    m = c * np.sum(w * x * target(x)) / Z
    V = c * np.sum(w * (x ** 2) * target(x)) / Z - m ** 2
    return Gaussian(m, V, Z=Z)

def display_fits(xs, target, methods, colors, legend, local=False):
    xs = vs.sample.x
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
    plt.legend(legend)
    target_xs = target(xs.squeeze())
    target_x = target(x.squeeze())
    if local:
        target_xs *= vs.sample.kernel(xs)
        target_x *= vs.sample.kernel(x)
    plt.stem(xs.squeeze(), target_xs, linefmt='k-', markerfmt='ko')
    plt.plot(x.squeeze(), target_x, 'k')
    plt.show()


target = ExponentialPowerLaw(beta=BETA)
v = target.V.squeeze()
h2 = 10 * v

s = Sample(target, 0, h2, ndraws=NPTS)
vs = VariationalFit(s, maxiter=10)
ds = VariationalFitIS(s)
bs = VariationalFitBMC(s, var=v)

gs_loc_fit = gauss_hermite(target, h2, 250)
gh_loc_fit = gauss_hermite(target, h2, NPTS)

print ('Error for VS: %f (expected: %f)'\
           % (gs_loc_fit.kl_div(vs.loc_fit), vs.kl_error))
print ('Error for IS: %f (expected: %f)'\
           % (gs_loc_fit.kl_div(ds.loc_fit), ds.kl_error))
print ('Error for BMC: %f'% gs_loc_fit.kl_div(bs.loc_fit))
print ('Error for GH: %f' % gs_loc_fit.kl_div(gh_loc_fit))


display_fits(vs.sample.x, target, (vs, ds, bs), 
             ('blue', 'orange', 'green'), ('VS', 'IS', 'BMC'))
display_fits(vs.sample.x, target, (vs, ds, bs),
             ('blue', 'orange', 'green'), ('VS', 'IS', 'BMC'),
             local=True)
