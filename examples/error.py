import numpy as np
import pylab as plt
from scipy.special.orthogonal import h_roots

from variational_sampler import VariationalSampler
from _toy_dist import ExponentialPowerLaw
from _display import display_fit

BETA = 3
DIM = 1
KIND = 'kl'

def error(v, kind='kl'):
    if kind == 'kl':
        return v.kl_error
    elif kind == 'integral':
        return v.var_integral[0, 0]
    elif kind == 'theta':
        return v.var_theta[0, 0]

target = ExponentialPowerLaw(beta=BETA, dim=DIM)

npts = 10 ** np.arange(1, 6)
e, e0 = [], []

for n in npts:
    vs = VariationalSampler(target, (np.zeros(DIM), target.V), n)
    f = vs.fit('kl')
    f0 = vs.fit('l')
    e.append(error(f, kind=KIND))
    e0.append(error(f0, kind=KIND))

e = np.array(e)
e0 = np.array(e0)

plt.figure()
plt.plot(npts, np.log(e))
plt.plot(npts, np.log(e0))
plt.legend(('VS', 'IS'))
plt.show()
