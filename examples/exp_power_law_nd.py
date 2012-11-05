import numpy as np
import pylab as plt
from variational_sampler.variational_sampler import VariationalFit
from variational_sampler.importance_sampler import ImportanceFit
from variational_sampler.gp_sampler import GaussianProcessFit
from variational_sampler.sampling import Sample
from variational_sampler.gaussian import Gaussian
from _toy_dist import ExponentialPowerLaw


BETA = 3
DIM = 5
NPTS = 10 * DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)
h2 = np.diagonal(target.V)

s = Sample((np.zeros(DIM), h2), ndraws=NPTS)
vs = VariationalFit(target, s)
v0 = ImportanceFit(target, s)
v1 = GaussianProcessFit(target, s, var=h2)
gopt = Gaussian(target.m, target.V, Z=target.Z)

print('Error for VS: %f (expected: %f)'\
          % (gopt.kl_div(vs.fit), vs.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gopt.kl_div(v0.fit), v0.kl_error))
print('Error for BMC: %f'% gopt.kl_div(v1.fit))
