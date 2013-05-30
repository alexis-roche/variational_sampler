import numpy as np
from scipy.optimize import fmin_ncg
import pylab as plt
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian, FactorGaussian
from variational_sampler.gaussian_mixture import GaussianMixture

DIM = 30
NDRAWS = (((DIM + 2) * (DIM + 1)) / 2) * np.array((2, 4, 8, 16))
REPEATS = 1

NCENTERS = 100
DISTANCE = 0
"""
p(x) = (1/N) sum_k g(xi, mi, Vi)

g(x) = K exp(-.5 * x.T x)
g'(x) = -g x
g"(x) = g x x.T - g I = g(xxT - I) 

log p(x) 
--> g / p
--> H / p - g g^T / p^2
"""

METHODS = ('kl', 'l', 'gp')
PARAMS = {
    'l': {},
    'kl': {'minimizer': 'quasi_newton'},
    'gp': {'var': 1}
}
COLORS = {
    'l': 'gray',
    'kl': 'black',
    'gp': 'black'
}
LINEFMT = {
    'l': 'o-',
    'kl': 'o-',
    'gp': 'o:'
}

mahalanobis = lambda f: np.sum(f.m * np.dot(np.linalg.inv(f.V), f.m))
MEASURES = lambda f, g: np.abs(f.Z - g.Z),\
    lambda f, g: np.sqrt(np.sum((f.m - g.m) ** 2)),\
    lambda f, g: np.sqrt(np.sum((f.V - g.V) ** 2)),\
    lambda f, g: kl_div(g, f)
get_measures = lambda f, g: np.array([m(f, g) for m in MEASURES])


def kl_div(g, f):
    try:
        d = g.kl_div(f)
    except:
        d = np.inf
    return d

def grad_gauss(g, x):
    dx = x - g.m
    gval = g(x.reshape((len(x), 1)))
    return -gval * dx


def hess_gauss(g, x):
    dx = x - g.m
    A = np.dot(dx.reshape((len(x), 1)), dx.reshape((1, len(x)))) - np.eye(len(x))
    gval = g(x.reshape((len(x), 1)))
    return gval * A


def grad_mixt(mixt, x):
    gx = np.array([grad_gauss(g, x) for g in mixt._gaussians])
    return np.sum(mixt._weights * gx.T, -1)


def grad_log_mixt(mixt, x):
    p = mixt(x.reshape((len(x), 1)))
    return grad_mixt(mixt, x) / p


def hess_mixt(mixt, x):
    Hx = np.array([hess_gauss(g, x) for g in mixt._gaussians])
    return np.sum(mixt._weights * Hx.T, -1)


def hess_log_mixt(mixt, x):
    p = mixt(x.reshape((len(x), 1)))
    v = grad_mixt(mixt, x) / p
    H = hess_mixt(mixt, x)
    return H / p - np.dot(v.reshape((len(x), 1)), v.reshape((1, len(x))))


def laplace(mixt):
    cost = lambda x: -np.log(mixt(x.reshape((len(x), 1))))
    grad = lambda x: -grad_log_mixt(mixt, x)
    hess = lambda x: -hess_log_mixt(mixt, x)
    m = fmin_ncg(cost, np.zeros(mixt.dim), grad, fhess=hess)
    V = np.linalg.inv(hess(m))
    K = np.exp(-cost(m))
    return Gaussian(m, V, K=K)


def display(a, measures, methods=METHODS):
    for k in range(len(MEASURES)):
        plt.figure()
        for i in range(len(methods)):
            m = methods[i]
            dat = measures[m][k]
            q3 = np.percentile(dat, 75, axis=1)
            q2 = np.percentile(dat, 50, axis=1)
            q1 = np.percentile(dat, 25, axis=1)
            err = np.concatenate(((q2-q1).reshape((1, len(q1))), (q3-q2).reshape((1, len(q1)))))
            plt.errorbar(a, q2, err, fmt=LINEFMT[m], color=COLORS[m])
    aux = measures['laplace'][k]
    a0, a1, _, _ = plt.axis()
    plt.plot((a0, a1), (aux, aux), 'k--')
    plt.xlabel('Sample size', fontsize=14)
    plt.ylabel('Excess KL divergence', fontsize=14)
    plt.show()


durations = {}
measures = {}

for m in METHODS:
    durations[m] = np.zeros((len(NDRAWS), REPEATS))
    measures[m] = np.zeros((len(MEASURES), len(NDRAWS), REPEATS))

scale = DISTANCE / np.sqrt(DIM)
print('scale = %f' % scale)

ok = False
while not ok:
    gaussians = [FactorGaussian(scale * np.random.normal(size=DIM), np.ones(DIM))\
                     for c in range(NCENTERS)]
    mixt = GaussianMixture((1 / float(NCENTERS)) * np.ones(NCENTERS), gaussians)
    target = lambda x: np.log(mixt(x))
    gs_fit = Gaussian(mixt.m, mixt.V, Z=mixt.Z)
    try:
        gl_fit = laplace(mixt)
        ok = True
    except:
        print('Weird distribution... reject')
measures['laplace'] = get_measures(gl_fit, gs_fit)

for i in range(len(NDRAWS)):
    ndraws = NDRAWS[i]
    for r in range(REPEATS):
        vs = VariationalSampler(target, (gl_fit.m, gl_fit.V), ndraws=ndraws)
        for m in METHODS:
            print(m)
            f = vs.fit(m, **PARAMS[m])
            durations[m][i, r] = f.time
            measures[m][:, i, r] = get_measures(f.fit, gs_fit)

display(NDRAWS, measures)

for m in METHODS:
    print('%s computation time: %s' % (m, np.median(durations[m], 1)))


