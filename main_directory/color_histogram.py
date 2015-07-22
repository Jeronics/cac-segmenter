import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def color_histogram(data):
    # Random gaussian data.


    # Plot histogram.
    n, bins, patches = plt.hist(data, 25, normed=1, color='green')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,pi]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    col = col * np.pi

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', matplotlib.cm.hsv(c))

    plt.show()


if __name__ == '__main__':
    Ntotal = 1000
    data = 0.05 * np.random.randn(Ntotal) + 0.5
    color_histogram(data, )