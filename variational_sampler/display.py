import numpy as np
import pylab as plt


def display_fit(xs, target, method, color=None, acronym=None,
                xmax=None, linewidth=2):
    if not hasattr(method, '__iter__'):
        method = (method, )
        if not color is None:
            color = (color, )
        if not acronym is None:
            acronym = (acronym, )
    if not xs.shape[0] == 1:
        raise ValueError('univariate data expected')
    if xmax == None:
        xmax = int(np.max(np.abs(xs.squeeze()))) + 1
    x = np.linspace(-xmax, xmax, 2 * xmax / 0.01)
    x = np.reshape(x, (1, x.size))
    fits = [m.fit(x) for m in method]
    if color is None:
        for fit in fits:
            plt.plot(x.squeeze(), fit, linewidth=linewidth)
    else:
        for fit, col in zip(fits, color):
            plt.plot(x.squeeze(), fit, col, linewidth=linewidth)
    if not acronym is None:
        plt.legend(acronym)
    target_xs = np.exp(target(xs.squeeze()))
    target_x = np.exp(target(x.squeeze()))
    plt.stem(xs.squeeze(), target_xs, linefmt='k-', markerfmt='ko',
             basefmt='k-')
    plt.plot(x.squeeze(), target_x, 'k')
    x0, x1, y0, y1 = plt.axis()
    plt.plot((x0, x1), (0, 0), 'k')
    plt.show()
