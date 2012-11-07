import numpy as np
from variational_sampler import VariationalSampler

DIM = 50

def target(x, beta=2):
    """
    Function that takes an array with shape (dim, n) as input and
    returns an array with shape (n,) that contains the corresponding
    target log-distribution values.
    """
    return np.sum(-.5 * np.abs(x) ** beta, 0)


"""
Tune the mean and variance of the sampling kernel. If we use as vector
as the variance, it will be understood as a diagonal matrix.
"""
ms = np.zeros(DIM)
vs = np.ones(DIM)

"""
Create a variational sampler object.
"""
v = VariationalSampler(target, (ms, vs), family='factor_gaussian',
                       ndraws=100 * DIM)

"""
Get the adjusted normalization constant, mean and variance.
"""
print('Estimated normalizing constant: %f' % v.fit.Z)
print('Estimated mean: %s' % v.fit.m)
print('Estimated variance (diagonal): %s' % v.fit.v)
