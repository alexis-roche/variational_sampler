import numpy as np
import pylab as plt
from variational_sampler.variational_sampler import (VariationalFit,
                                                     StraightFit)
from variational_sampler.gaussian_process import GaussianProcessFit
from variational_sampler.sampling import Sample
from variational_sampler.toy_examples import ExponentialPowerLaw


BETA = 2
DIM = 20
NPTS = DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)
h2 = np.diagonal(target.V)

s = Sample((np.zeros(DIM), h2), kernel='match', ndraws=NPTS)
vf = VariationalFit(target, s)
sf = StraightFit(target, s)
gf = GaussianProcessFit(target, s, var=h2)
