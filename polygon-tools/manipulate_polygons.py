__author__ = 'jeroni'
import turning_function as tf
import numpy as np

def translate_polygon(poly, t_x,t_y):
    translation=np.tile(np.array([t_x,t_y]),[len(poly),1])
    t_poly = poly +translation
    return t_poly

def scale_polygon(poly, scalar):
    s_poly= poly*scalar
    return s_poly

def rotate_polygon(poly, rot):
    rotation_matrix=np.array([[np.cos(rot), -np.sin(rot)],[np.sin(rot), np.cos(rot)]])
    r_poly=np.dot(poly, rotation_matrix)
    return r_poly

if __name__=='__main__':
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
    poly_2 = translate_polygon(poly_1, 1,2)
    poly_3 = scale_polygon(poly_1,10)
    poly_4 = rotate_polygon(poly_1, np.pi/4.)
    print poly_3
    tf.plot_polygon(poly_1)
    tf.plot_polygon(poly_2)
    tf.plot_polygon(poly_3)
    tf.plot_polygon(poly_4)
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
    x_2, angles_2 = tf.turning_function(poly_2, plot_func=False)

