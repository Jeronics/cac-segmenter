__author__ = 'jeroni'
# http://www.cs.cornell.edu/~dph/papers/achkm-tpami-91.pdf

# OTHER IDEAS: http://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK5/NODE196.HTM
# https://www.student.cs.uwaterloo.ca/~cs763/Projects/phil.pdf
import math
import numpy as np
import collections
from matplotlib import pyplot as plt
import manipulate_polygons as mp


def multiple_euclid_dist(p1, p2):
    dis = p1 - p2
    return [np.sqrt(np.dot(x, x)) for x in dis]


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def multiple_angles(v1, v2):
    '''
    Calculates the angle of all vectors of v_list with respect to the vectors in v2

    :param v_list:
    :param vector:
    :return:
    '''

    return [angle(a, b) for a, b in zip(v1, v2)]


def angle(v1, v2, sign=1):
    # if np.sin(dotproduct(v1, v2) / (length(v1) * length(v2))) == 1.0:
    # return np.pi / 2.
    # elif np.sin(dotproduct(v1, v2) / (length(v1) * length(v2))) == -1.0:
    # return np.pi * 3 / 2.
    # else:
    # signed_angle = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])
    # print dotproduct(v1, v2) / (length(v1) * length(v2))
    return sign * math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def radiants_to_degrees(radiant):
    return radiant * 180 / np.pi


def get_value(x, val, f_val):
    for a, b in zip(val, f_val):
        if x <= a:
            return b
    return b


def U(x, val, f_val):
    return np.array([get_value(i, val, f_val) for i in x])


def plot_discrete_funct(val, f_val, display=True, label_points=False, plot_fig=True):
    N = 1000
    x = np.linspace(0, val[-1], N)
    if plot_fig:
        plt.plot(x, U(x, val, f_val))
        plt.xlim(0, val[-1])
        plt.ylim(int(min(min(f_val), 0) - 0.9), int(max(f_val) + 1))
        plt.xticks(np.append([0], val), ['p_' + str(i) for i, v in enumerate(np.append([0], val))])
        if display:
            plt.show()



def plot_polygons(p, q):
    plot_polygon(p, display=False)
    plot_polygon(q, display=False)
    plt.show()


def turn(v1, v2):
    '''
    https://sites.google.com/site/turningfunctions/

    :param v1:
    :param v2:
    :return:
    '''
    z_ = np.zeros([len(v1), 1])
    v1_ = np.append(v1, z_, axis=1)
    v2_ = np.append(v2, z_, axis=1)
    sign = [0 if np.cross(a, b)[-1] == 0 else 1 if np.cross(a, b)[-1] > 0 else -1 for a, b in zip(v2_, v1_)]
    angles = [angle(a, b, sign=s) for a, b, s in zip(v2_, v1_, sign)]
    return angles


def plot_polygon(p, display=True, fig_title=''):
    '''
    This function plots the polygon with the turning function
    :param p:
    :return:
    '''
    x, angles = turning_function(p, plot_func=False)

    fig = plt.figure(figsize=(20, 5))
    plt.suptitle(fig_title)
    ax = fig.add_subplot(121)
    # Plots the polygon
    # plt.subplot(1, 2, 1)
    p_ = np.append(p, [p[0]], axis=0)
    plt.title('Polygon')
    plt.axis('equal')
    plt.xlim(int(min(p[:, 0] - 1.01)), int(max(p[:, 0] + 1.01)))
    plt.ylim(int(min(p[:, 1] - 1.01)), int(max(p[:, 1] + 1.01)))
    plt.plot(np.transpose(p_)[0], np.transpose(p_)[1])
    for idx, (i, j) in enumerate(p):
        ax.annotate('p_%s' % idx, xy=(i, j), textcoords='offset points')  # <--

    # Plots the turning function
    fig.add_subplot(122)
    plot_discrete_funct(x, angles, display=False)
    plt.title('Turning function')
    plt.xlabel('Perimeter distance')
    plt.ylabel('Turning angle')
    if display:
        plt.show()


def turning_function(p, plot_func=True):
    q = collections.deque(p)
    q.rotate(-1)
    v1 = p - q
    v2 = collections.deque(v1)
    v2.rotate(1)

    angles = turn(v2, v1)
    x = multiple_euclid_dist(p, q)

    angles = np.cumsum(angles)
    x = np.cumsum(x)
    x /= float(x[-1])
    if plot_func:
        plot_discrete_funct(x, angles)
    return x, angles


if __name__ == '__main__':
    poly_1 = np.array([
        [5, 3],
        [4, 0],
        [0, 1],
        [2, 2.9],
        [1.0, 5],
    ])
    # turning_function(poly_1)
    # plot_polygon(poly_1)

    poly_1 = np.array([
        [0, 0],
        [0, 3],
        [1, 3],
        [1, 0],
    ])
    # turning_function(poly_1)
    poly_1 = np.array([
        [5, 3],
        [4, 0],
        [0, 1],
        [2, 2.9],
        [1.0, 5],
    ])
    # turning_function(poly_1)
    poly_1 = np.array([
        [0, 0],
        [0, 2],
        [6, 2],
        [6, 0],
        [5, 0],
        [5, 1],
        [4, 1],
        [4, 0],
    ])
    # turning_function(poly_1)
    plot_polygon(poly_1)
