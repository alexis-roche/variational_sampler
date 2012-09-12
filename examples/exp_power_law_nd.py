import numpy as np
import pylab as plt
from variational_sampler import (VariationalFit,
                                 DirectFit,
                                 Gaussian,
                                 Sample)
from variational_sampler.toy_examples import ExponentialPowerLaw


BETA = 2
DIM = 20
NPTS = DIM ** 2

target = ExponentialPowerLaw(beta=BETA, dim=DIM)
h2 = target.V

s = Sample(target, np.zeros(DIM), h2, ndraws=NPTS)
vs = VariationalFit(s, minimizer='conjugate', tol=1e-2)
ds = DirectFit(s)

