__author__ = 'jeroni'
import turning_function as tf
import numpy as np


def translate_polygon(poly, t_x, t_y):
    translation = np.tile(np.array([t_x, t_y]), [len(poly), 1])
    t_poly = poly + translation
    return t_poly


def scale_polygon(poly, scalar):
    s_poly = poly * scalar
    return s_poly


def rotate_polygon(poly, rot):
    rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    r_poly = np.dot(poly, rotation_matrix)
    return r_poly


def distort_polygon(poly, sigma):
    mu = 0
    s = np.random.normal(mu, sigma, [len(poly), 2])
    d_poly = poly + s
    return d_poly


if __name__ == '__main__':
    poly_1 = np.array([
        [0, 0],
        [0, 2],
        [6, 2],
        [6, 0],
        [4, 0],
        [4, 1],
        [3, 1],
        [3, 0],
    ])
    poly_2 = translate_polygon(poly_1, 1, 2)
    poly_3 = scale_polygon(poly_1, 10)
    poly_4 = rotate_polygon(poly_1, np.pi / 4.)
    poly_5 = distort_polygon(poly_1, 0.1)
    tf.plot_polygon(poly_1)
    tf.plot_polygon(poly_2)
    tf.plot_polygon(poly_3)
    tf.plot_polygon(poly_4)
    tf.plot_polygon(poly_5)
    poly_2 = np.array([
        [0, 0],
        [0, 2],
        [6, 2],
        [6, 0],
        [5, 0],
        [5, 1],
        [4, 1],
        [4, 0],
    ])
    x_1, angles_1 = tf.turning_function(poly_1, plot_func=False)
    x_2, angles_2 = tf.turning_function(poly_1, plot_func=False)
