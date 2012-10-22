import numpy as np
import pylab as plt
from variational_sampler.variational_sampler import VariationalFit
from variational_sampler.importance_sampler import ImportanceFit
from variational_sampler.gp_sampler import GaussianProcessFit
from variational_sampler.sampling import Sample
from _toy_dist import ExponentialPowerLaw


BETA = 2
DIM = 20
NPTS = DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)
h2 = np.diagonal(target.V)

s = Sample((np.zeros(DIM), h2), context='kernel', ndraws=NPTS)
vf = VariationalFit(target, s)
sf = ImportanceFit(target, s)
gf = GaussianProcessFit(target, s, var=h2)
