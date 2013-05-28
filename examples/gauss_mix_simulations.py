import numpy as np
import pylab as plt
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian, FactorGaussian
from variational_sampler.gaussian_mixture import GaussianMixture

DIM = 3
NDRAWS = (((DIM + 2) * (DIM + 1)) / 2) * np.array((2, 4, 8, 16))
"""
NDRAWS = (((DIM + 2) * (DIM + 1)) / 2) * 4
"""
REPEATS = 5
SCALING = 1

NCENTERS = 10
DISTANCE = 1

METHODS = {
    'l': {},
    'kl': {'minimizer': 'quasi_newton'},
    'gp': {'var': 1}
}
COLORS = {
    'l': 'orange',
    'kl': 'blue',
    'gp': 'green'
}
mahalanobis = lambda f: np.sum(f.m * np.dot(np.linalg.inv(f.V), f.m))
MEASURES = lambda f, g: np.abs(f.Z - g.Z),\
    lambda f, g: np.sqrt(np.sum((f.m - g.m) ** 2)),\
    lambda f, g: np.sqrt(np.sum((f.V - g.V) ** 2)),\
    lambda f, g: g.kl_div(f)
get_measures = lambda f, g: np.array([m(f, g) for m in MEASURES])

def display(a, measures, robust=False):
    if robust:
        mu = lambda x: np.median(x, 1)
        std = lambda x: 1.4826 * np.median(np.abs(x.T - mu(x)), 0)
    else:
        mu = lambda x: np.mean(x, 1)
        std = lambda x: np.std(x, 1)
    for k in range(len(MEASURES)):
        plt.figure()
        for m in METHODS.keys():
            plt.errorbar(a, mu(measures[m][k]), std(measures[m][k]), fmt='o-', color=COLORS[m])
        plt.legend(METHODS.keys(), loc=0)
    plt.show()


durations = {}
measures = {}

for m in METHODS.keys():
    durations[m] = np.zeros((len(NDRAWS), REPEATS))
    measures[m] = np.zeros((len(MEASURES), len(NDRAWS), REPEATS))

scale = DISTANCE / np.sqrt(DIM)
print('scale = %f' % scale)

gaussians = [FactorGaussian(scale * np.random.normal(size=DIM), np.ones(DIM))\
                         for c in range(NCENTERS)]
mixt = GaussianMixture((1 / float(NCENTERS)) * np.ones(NCENTERS), gaussians)
target = lambda x: np.log(mixt(x))
gs_fit = Gaussian(mixt.m, mixt.V, Z=mixt.Z)
kernel = (mixt.m, SCALING * mixt.V)
        
for i in range(len(NDRAWS)):
    ndraws = NDRAWS[i]
    for r in range(REPEATS):
        vs = VariationalSampler(target, kernel, ndraws=ndraws)
        for m in METHODS.keys():
            print(m)
            f = vs.fit(m, **METHODS[m])
            durations[m][i, r] = vs.sampling_time + f.time
            measures[m][:, i, r] = get_measures(f.fit, gs_fit)

display(NDRAWS, measures)

for m in METHODS.keys():
    print('%s computation time: %s' % (m, np.median(durations[m], 1)))
