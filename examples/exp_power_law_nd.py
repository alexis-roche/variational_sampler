import numpy as np
import pylab as plt
from variational_sampler.variational_sampler import (VariationalFit,
                                                     ImportanceFit)
from variational_sampler.sampling import Sample
from variational_sampler.toy_examples import ExponentialPowerLaw


BETA = 2
DIM = 50
NPTS = DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)
h2 = np.diagonal(target.V)

s = Sample(target, np.zeros(DIM), h2, ndraws=NPTS)
vs = VariationalFit(s, minimizer='quasi_newton')
ds = ImportanceFit(s)
