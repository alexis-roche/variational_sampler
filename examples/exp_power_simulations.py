import numpy as np
import pylab as plt
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian
from _toy_dist import ExponentialPowerLaw

BETA = 3
DIM = 5
NDRAWS = (((DIM + 2) * (DIM + 1)) / 2) * np.array((5, 10, 20, 50))
SCALING = 1
REPEATS = 200

TARGET = ExponentialPowerLaw(beta=BETA, dim=DIM)
H2 = SCALING * np.diagonal(TARGET.V)
GS_FIT = Gaussian(TARGET.m, TARGET.V, Z=TARGET.Z)
mahalanobis = lambda f: np.sum(f.m * np.dot(np.linalg.inv(f.V), f.m))

MEASURES = lambda f: f.Z,\
    lambda f: (f.Z - TARGET.Z) ** 2,\
    lambda f: mahalanobis(f),\
    lambda f: GS_FIT.kl_div(f)

get_measures = lambda f: np.array([m(f) for m in MEASURES])

def display(durations, measures, robust=False):
    if robust:
        mu = lambda x: np.median(x, 1)
        std = lambda x: 1.4826 * np.median(np.abs(x.T - mu(x)), 0)
    else:
        mu = lambda x: np.mean(x, 1)
        std = lambda x: np.std(x, 1)
    for k in range(len(MEASURES)):
        plt.figure()
        plt.errorbar(NDRAWS, mu(measures['l'][k]), std(measures['l'][k]), fmt='o-', color='orange')
        plt.errorbar(NDRAWS, mu(measures['kl'][k]), std(measures['kl'][k]), fmt='o-', color='red')
    plt.show()

durations = {
    'l': np.zeros((len(NDRAWS), REPEATS)),
    'kl': np.zeros((len(NDRAWS), REPEATS))
}
measures = {
    'l': np.zeros((len(MEASURES), len(NDRAWS), REPEATS)),
    'kl': np.zeros((len(MEASURES), len(NDRAWS), REPEATS))
}

for i in range(len(NDRAWS)):
    ndraws = NDRAWS[i]
    for r in range(REPEATS):
        vs = VariationalSampler(TARGET, (np.zeros(DIM), H2), ndraws=ndraws)
        f = vs.fit('l')
        durations['l'][i, r] = vs.sampling_time + f.time
        measures['l'][:, i, r] = get_measures(f.fit)
        #f = vs.fit('kl', family='factor_gaussian')
        f = vs.fit('kl', minimizer='quasi_newton')
        durations['kl'][i, r] = vs.sampling_time + f.time
        measures['kl'][:, i, r] = get_measures(f.fit)
    


