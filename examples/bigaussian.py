import numpy as np
import pylab as plt
from variational_sampler import VariationalSampler
from _display import display_fit

def target(x):
    return np.exp(-.5 * (x + 2) ** 2) + np.exp(-.5 * (x - 2) ** 2)

v1 = VariationalSampler(target, (-4, .1), context='kernel', ndraws=100)
v2 = VariationalSampler(target, (4, .1), context='kernel', ndraws=100)
v3 = VariationalSampler(target, (1, .1), context='kernel', ndraws=100)

plt.figure()
x = np.linspace(-6, 6, num=100)
plt.plot(x, target(x), 'k', linewidth=2)
plt.plot(x, v1.fit(x), 'b')
plt.plot(x, v2.fit(x), 'r')
plt.plot(x, v3.fit(x), 'g')
plt.plot(x, v1.sample.kernel(x), 'b:')
plt.plot(x, v2.sample.kernel(x), 'r:')
plt.plot(x, v3.sample.kernel(x), 'g:')
plt.show()

