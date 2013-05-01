import numpy as np
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian
from _toy_dist import ExponentialPowerLaw


BETA = 3
DIM = 5
NPTS = 10 * DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)

h2 = np.diagonal(target.V)

vs = VariationalSampler(target, (np.zeros(DIM), h2), ndraws=NPTS)
f = vs.fit()
f0 = vs.fit('l')
f2 = vs.fit('gp', var=h2)
gopt = Gaussian(target.m, target.V, Z=target.Z)

print('Error for VS: %f (expected: %f)'\
          % (gopt.kl_div(f.fit), f.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gopt.kl_div(f0.fit), f0.kl_error))
print('Error for BMC: %f' % gopt.kl_div(f2.fit))
