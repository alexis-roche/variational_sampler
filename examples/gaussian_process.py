import numpy as np
from variational_sampler import VariationalSampler

DIM = 2
SIGMA = 0.1

xt = np.linspace(0, 1, num=DIM)
y = xt + SIGMA * np.random.normal(size=DIM)
y = 0 * xt


A = np.eye(DIM)
for i in range(1, DIM):
    A[i, i-1] = -1
Ainv = np.linalg.inv(A)
prior = (np.zeros(DIM), np.dot(A, A.T))

def likelihood(x):
    """
    x is an array with shape (d, N)
    """
    tmp = np.abs((x.T - y).T).sum(0)
    return np.exp(-tmp / SIGMA)

v = VariationalSampler(likelihood, prior,
                       context='kernel',
                       ndraws=1000000)

"""
v2 = VariationalSampler(likelihood, prior,
                       context='kernel',
                       family='factor_gaussian',
                       ndraws=1000)

"""
