import numpy as np
import pylab as plt


def display_fit(xs, target, method, color=None, linefmt=None, acronym=None, xmax=None):
    if not hasattr(method, '__iter__'):
        method = (method, )
        if not color is None:
            color = (color, )
        if not linefmt is None:
            linefmt = (linefmt, )
        if not acronym is None:
            acronym = (acronym, )
    if not xs.shape[0] == 1:
        raise ValueError('univariate data expected')
    if xmax == None:
        xmax = int(np.max(np.abs(xs.squeeze()))) + 1
    x = np.linspace(-xmax, xmax, 2 * xmax / 0.01)
    x = np.reshape(x, (1, x.size))
    fits = [m(x) for m in method]
    for fit, col, fmt in zip(fits, color, linefmt):
        plt.plot(x.squeeze(), fit, fmt, color=col, linewidth=2)
    if not acronym is None:
        plt.legend(acronym)
    target_xs = np.exp(target(xs.squeeze()))
    target_x = np.exp(target(x.squeeze()))
    plt.stem(xs.squeeze(), target_xs, linefmt='k-', markerfmt='ko')
    plt.plot(x.squeeze(), target_x, 'k')
    plt.plot((-xmax, xmax), (0, 0), 'k')
    plt.show()
