import numpy as np
from variational_sampler import VariationalSampler
from variational_sampler.gaussian import Gaussian, FactorGaussian
from variational_sampler.gaussian_mixture import GaussianMixture

DIM = 5
NCENTERS = 10
DISTANCE = .2
NPTS = 10 * DIM ** 2

gaussians = [FactorGaussian((DISTANCE / np.sqrt(DIM)) * np.random.normal(size=DIM),
                            np.ones(DIM))\
                               for i in range(NCENTERS)]
mixture = GaussianMixture((1 / float(NCENTERS)) * np.ones(NCENTERS), gaussians)
target = lambda x: np.log(mixture(x))

vs = VariationalSampler(target, (mixture.m, mixture.V), ndraws=NPTS)
f = vs.fit()
f0 = vs.fit('l')
f2 = vs.fit('gp', var=1)
gopt = Gaussian(mixture.m, mixture.V, Z=mixture.Z)

print('Error for VS: %f (expected: %f)'\
          % (gopt.kl_div(f.fit), f.kl_error))
print('Error for IS: %f (expected: %f)'\
           % (gopt.kl_div(f0.fit), f0.kl_error))
print('Error for BMC: %f' % gopt.kl_div(f2.fit))

