__author__ = 'jeroni'
# http://www.cs.cornell.edu/~dph/papers/achkm-tpami-91.pdf
import math
import numpy as np
import collections
from matplotlib import pyplot as plt


def multiple_euclid_dist(p1, p2):
    dis = p1 - p2
    return [np.sqrt(np.dot(x, x)) for x in dis]


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def multiple_angles(v_list, vector):
    '''
    Calculates the angle of all vectors of v_list with respect to the vector

    :param v_list:
    :param vector:
    :return:
    '''
    return [angle(v, vector) for v in v_list]


def angle(v1, v2):
    print v1, v2
    if np.sin(dotproduct(v1, v2) / (length(v1) * length(v2))) == 1.0:
        return np.pi / 2.
    elif np.sin(dotproduct(v1, v2) / (length(v1) * length(v2))) == -1.0:
        return np.pi * 3 / 2.
    else:
        return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def radiants_to_degrees(radiant):
    return radiant * 180 / np.pi


# def plot_discontinuous_funct(x,y, display=True):
# def heaviside(x):
# """See http://stackoverflow.com/a/15122658/554319"""
# y = 0.5 * (np.sign(x) + 1)
#         y[np.diff(y) >= 0.5] = np.nan
#         return y
#
#     def U(x, n):
#         aux= sum([heaviside(x - 1 / float(i)) / i ** 2 for i in range(1, n)])
#         print aux
#         return aux
#
#     N = 1000
#     x = np.linspace(0, 1, N)
#     plt.plot(x, U(x, 50))
#     if display:
#         plt.show()


def plot_discontinuous_funct(val, f_val, display=True):
    def get_value(x):
        for a, b in zip(val, f_val):
            if x <= a:
                return b
    def U(x):
        return [get_value(i) for i in x]

    N = 1000
    x = np.linspace(0, val[-1], N)
    print len(x), len(U(x))
    plt.plot(x, U(x))
    if display:
        plt.xlim(0, val[-1])
        print 'value', val[-1]
        plt.show()


def turning_funcion(p):
    q = collections.deque(p)
    q.rotate(-1)
    vectors = p - q
    x_axis = np.array([1, 0])
    angles = multiple_angles(vectors, x_axis)
    x = multiple_euclid_dist(p, q)
    print x
    x = np.cumsum(x)
    x=x/float(x[-1])
    print x
    print len(angles)
    plot_discontinuous_funct(x, angles)
    plt.show()

    return 0


if __name__ == '__main__':
    poly_1 = np.array([
        [0, 0],
        [0, 2],
        [1, 0]
    ])
    turning_funcion(poly_1)


