import numpy as np
import pylab as plt
from variational_sampler import VariationalSampler
from _display import display_fit

def dist(x):
    return np.exp(-.5 * (x + 2) ** 2) + np.exp(-.5 * (x - 2) ** 2)

def logdist(x):
    return np.log(dist(x))

v1 = VariationalSampler(logdist, (-4, .1), context='kernel', ndraws=100)
v2 = VariationalSampler(logdist, (4, .1), context='kernel', ndraws=100)
v3 = VariationalSampler(logdist, (1, .1), context='kernel', ndraws=100)

plt.figure()
x = np.linspace(-6, 6, num=100)
plt.plot(x, dist(x), 'k', linewidth=2)
plt.plot(x, v1.fit(x), 'b')
plt.plot(x, v2.fit(x), 'r')
plt.plot(x, v3.fit(x), 'g')
plt.plot(x, v1.sample.kernel(x), 'b:')
plt.plot(x, v2.sample.kernel(x), 'r:')
plt.plot(x, v3.sample.kernel(x), 'g:')
plt.show()

