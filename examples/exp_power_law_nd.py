import numpy as np
import pylab as plt
from variational_sampler.variational_sampler import (VariationalFit,
                                                     ClassicalFit)
from variational_sampler.bayesian_monte_carlo import BayesianMonteCarloFit
from variational_sampler.sampling import Sample
from variational_sampler.toy_examples import ExponentialPowerLaw


BETA = 2
DIM = 20
NPTS = DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)
h2 = np.diagonal(target.V)

s = Sample(target, np.zeros(DIM), h2, ndraws=NPTS)
vs = VariationalFit(s)
ds = ClassicalFit(s)
bs = BayesianMonteCarloFit(s, var=h2)
