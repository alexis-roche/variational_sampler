import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler import VariationalSampler
from _toy_dist import ExponentialPowerLaw
from _display import display_fit

BETA = 3
DIM = 2
KIND = 'kl'

def error(v, kind='kl'):
    if kind == 'kl':
        return v.kl_error
    elif kind == 'moment':
        return v.var_moment[0, 0]
    elif kind == 'theta':
        return v.var_theta[0, 0]

target = ExponentialPowerLaw(beta=BETA, dim=DIM)

npts = 10 ** np.arange(1, 6)
e_vs, e_v0 = [], []

for n in npts:
    s = VariationalSampler(target, (np.zeros(DIM), target.V), n)
    vs = s.fit('kl')
    v0 = s.fit('naive_kl')
    e_vs.append(error(vs, kind=KIND))
    e_v0.append(error(v0, kind=KIND))

e_vs = np.array(e_vs)
e_v0 = np.array(e_v0)

plt.figure()
plt.plot(npts, np.log(e_vs))
plt.plot(npts, np.log(e_v0))
plt.legend(('VS', 'IS'))
plt.show()
