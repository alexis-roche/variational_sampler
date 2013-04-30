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

METHODS = {
    'l': {},
    'kl': {'minimizer': 'quasi_newton'},
    'gp': {'var': .1 * np.mean(H2)}
}
METHODS = {
    'l': {},
    'kl': {'minimizer': 'quasi_newton'}
}

mahalanobis = lambda f: np.sum(f.m * np.dot(np.linalg.inv(f.V), f.m))
MEASURES = lambda f: f.Z,\
    lambda f: f.Z - TARGET.Z,\
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
        for m in METHODS.keys():
            """plt.errorbar(NDRAWS, mu(measures[m][k]), std(measures[m][k]), fmt='o-')
            """
            plt.errorbar(mu(durations[m]), mu(measures[m][k]), std(measures[m][k]), fmt='o-')
        plt.legend(METHODS.keys(), loc=0)
    plt.show()

durations = {}
measures = {}
for m in METHODS.keys():
    durations[m] = np.zeros((len(NDRAWS), REPEATS))
    measures[m] = np.zeros((len(MEASURES), len(NDRAWS), REPEATS))

for i in range(len(NDRAWS)):
    ndraws = NDRAWS[i]
    for r in range(REPEATS):
        vs = VariationalSampler(TARGET, (np.zeros(DIM), H2), ndraws=ndraws)
        for m in METHODS.keys():
            print(m)
            f = vs.fit(m, **METHODS[m])
            durations[m][i, r] = vs.sampling_time + f.time
            measures[m][:, i, r] = get_measures(f.fit)


