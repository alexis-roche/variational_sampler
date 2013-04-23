import numpy as np
import pylab as plt
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian
from _toy_dist import ExponentialPowerLaw

DIM = 5
NPTS = 100 * DIM ** 2


def random_var():
    A = np.random.rand(DIM, DIM)
    return np.dot(A, A.T)


def target(x):
    return np.sum(-.5 * x * np.dot(INV_VAR, x), 0)


MU = np.zeros(DIM)
VAR = random_var()
INV_VAR = np.linalg.inv(VAR)
Z = np.sqrt((2 * np.pi) ** DIM * np.linalg.det(VAR))

gopt = Gaussian(MU, VAR, Z=Z)
vs = VariationalSampler(target, (MU, VAR), ndraws=NPTS)
f = vs.fit()
f0 = vs.fit('naive_kl')


print('Error for VS: %f (expected: %f)'\
          % (gopt.kl_div(f.fit), f.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gopt.kl_div(f0.fit), f0.kl_error))
