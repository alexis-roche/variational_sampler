import numpy as np
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian
from _toy_dist import ExponentialPowerLaw


BETA = 3
DIM = 15
NPTS = 8 * DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)

h2 = np.diagonal(target.V)

vs = VariationalSampler(target, (1., np.zeros(DIM), h2), ndraws=NPTS)
f = vs.fit(minimizer='quasi_newton')
f2 = vs.fit('kl2')
f0 = vs.fit('l')
#f2 = vs.fit('gp', var=h2)

gopt = Gaussian(target.m, target.V, Z=target.Z)

print('VS: error=%f (expected=%f), fitting time=%f'\
          % (gopt.kl_div(f.fit), f.kl_error, f.time))
print('VS2: error=%f (expected=%f), fitting time=%f'\
          % (gopt.kl_div(f2.fit), f2.kl_error, f2.time))
print('IS: error=%f (expected=%f), fitting time=%f'\
          % (gopt.kl_div(f0.fit), f0.kl_error, f0.time))
#print('Error for BMC: %f' % gopt.kl_div(f2.fit))
