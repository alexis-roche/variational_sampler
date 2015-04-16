import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian
from variational_sampler.toy_dist import ExponentialPowerLaw
from variational_sampler.display import display_fit

BETA = 2
NPTS = 5
DV = 4
DM = -2
XMAX = 8

def gauss_hermite_rule(npts, mk, vk):
    """
    Compute the points and weights for the Gauss-Hermite quadrature
    with the normalized Gaussian N(0, h2) as a weighting function.
    """
    x, w = h_roots(npts)
    x *= np.sqrt(2 * vk)
    x += mk
    w /= np.sqrt(np.pi)
    return x, w


target = ExponentialPowerLaw(beta=BETA)
v = float(target.V)
mk = target.m + DM
vk = DV * v

gs_fit = Gaussian(target.m, target.V, Z=target.Z)

# Random sampling approach
vs = VariationalSampler(target, (mk, vk), NPTS)
f_kl = vs.fit()
f_l = vs.fit('l')
f_gp = vs.fit('gp', var=v)

# Deterministic sampling approch (tweak a vs object)
x, w = gauss_hermite_rule(NPTS, mk, vk)
vsd = VariationalSampler(target, (mk, vk), NPTS, x=x, w=w)
fd_kl = vsd.fit()
fd_l = vsd.fit('l')
fd_gp = vsd.fit('gp', var=v)


print('Error for VS: %f (expected: %f)'\
          % (gs_fit.kl_div(f_kl.fit), f_kl.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gs_fit.kl_div(f_l.fit), f_l.kl_error))
print('Error for BMC: %f' % gs_fit.kl_div(f_gp.fit))
print('Error for GH: %f' % gs_fit.kl_div(fd_l.fit))
print('Error for VSd: %f' % gs_fit.kl_div(fd_kl.fit))
print('Error for GP: %f' % gs_fit.kl_div(fd_gp.fit))

acronyms = ('VS', 'IS', 'BMC')
colors = ('blue', 'red', 'green')
legend = ('VS', 'direct', 'spline')
plt.figure()
display_fit(vs.x, target, (f_kl, f_l, f_gp), colors, legend, xmax=XMAX)
plt.figure()
display_fit(vsd.x, target, (fd_kl, fd_l, fd_gp), colors, legend, xmax=XMAX)
