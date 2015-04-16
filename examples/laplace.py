import numpy as np
import pylab as plt

from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian
from variational_sampler.toy_dist import ExponentialPowerLaw
from variational_sampler.display import display_fit

DIM = 1
NPTS = 100
DM = 2

target = ExponentialPowerLaw(beta=1, dim=DIM)
vs = VariationalSampler(target, (DM + target.m, target.V), NPTS)
f = vs.fit().fit
fl = vs.fit('l').fit

context = Gaussian(DM + target.m, 2 * target.V)
target2 = lambda x: target(x) + context.log(x)
vs2 = VariationalSampler(target2, context, NPTS)
f2 = vs2.fit().fit / context
fl2 = vs2.fit('l').fit / context

if DIM == 1:
    display_fit(vs.x, target, (f, f2, fl, fl2),
                ('blue', 'green', 'orange', 'red'), 
                ('VS', 'VSc', 'IS', 'ISc'))

gopt = Gaussian(target.m, target.V, Z=target.Z)
print('Error for VS: %f' % gopt.kl_div(f))
print('Error for VSc: %f' % gopt.kl_div(f2))
print('Error for IS: %f' % gopt.kl_div(fl))
print('Error for ISc: %f' % gopt.kl_div(fl2))
