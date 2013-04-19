import numpy as np
from variational_sampler import VariationalSampler

DIM = 2
SIGMA = 0.1
POWER = 5

xt = np.linspace(0, 1, num=DIM)
y = xt + SIGMA * np.random.normal(size=DIM)
y = 0 * xt


A = np.eye(DIM)
for i in range(1, DIM):
    A[i, i-1] = -1
Ainv = np.linalg.inv(A)
prior = (np.zeros(DIM), np.dot(A, A.T))

def loglikelihood(x):
    """
    x is an array with shape (d, N)
    """
    tmp = (np.abs((x.T - y).T) ** POWER).sum(0)
    return -tmp / SIGMA

v = VariationalSampler(loglikelihood, prior,
                       context='kernel',
                       ndraws=1000000)
f = v.fit()
f2 = v.fit(family='factor_gaussian')

