__author__ = 'jeroni'
import turning_function as tf
import numpy as np
import collections
import sets
import manipulate_polygons as mp



# p1 = Point(0, 0)
# p2 = Point(1, 0)
# p3 = Point(1, 1)
# p4 = Point(0, 1)
#
# x = Polygon(p1, p2, p3, p4)
# print 'Area', x.area, x.perimeter
# print x.centroid
#
# print '\n'

def get_angle(input_val):
    str_val = str(input_val)
    val_array = str_val.replace('acos(', '')[:-1]
    prod = val_array.split('*')
    if len(prod) > 1:
        div = prod[1].split('/')
        root = div[0].replace('sqrt(', '').replace(')', '')
        angle = math.pow(np.cos(int(prod[0]) * np.sqrt(int(root)) / float(int(div[1]))), -1)
    else:
        if 'pi/' in str_val:
            denom = str_val.replace('pi/', '')
            angle = np.pi / float(int(denom))
        else:
            angle = input_val
    return angle


# agls = []
#
# for p, a in x.angles.items():
# print 'angle', a
# agls.append(get_angle(a))
# print '\n'
# print agls
# print sum(agls)

# p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
# poly = Polygon(p1, p2, p3, p4)
# print poly.angles[p1]

def integral_of_discrte_function(x, y, interval=[0, 1]):
    x_ = np.append(interval[0], x[:-1])
    x_ = x_.flatten()
    lengths = x - x_
    integral = sum([a * b for a, b in zip(lengths, y)])
    return integral


def shift_polygon(x, angles, i):
    s = x[i - 1] if i != 0 else 0
    x_shift = np.append(x[i:], x[:i] + x[-1]) - s
    f = angles[i - 1] if i != 0 else 0
    f_shift = np.append(angles[i:], angles[:i] + angles[-1]) - f
    return x_shift, f_shift


def subtract_histograms(x_a, angles_a, x_b, angles_b):
    set_a = sets.Set(x_a)
    set_b = sets.Set(x_b)
    x_merge = sorted(set_a.union(set_b))

    f_merge_a = tf.U(x_merge, x_a, angles_a)
    f_merge_b = tf.U(x_merge, x_b, angles_b)
    f_difference = f_merge_a - f_merge_b
    tf.plot_discrete_funct(x_merge, f_difference, plot_fig=False)
    angles_diff_sq = np.power(f_difference, 2)

    integral = integral_of_discrte_function(x_merge, angles_diff_sq, interval=[0, 1])
    return integral


def polygon_comparison(poly_a, poly_b):
    x_a, angles_a = tf.turning_function(poly_a, plot_func=False)
    x_b, angles_b = tf.turning_function(poly_b, plot_func=False)
    minimum_difference = max(max(abs(angles_a)), max(abs(angles_b)))

    for i, a in enumerate(x_a):
        x_a_s, angles_a_s = shift_polygon(x_a, angles_a, i)
        # tf.plot_discrete_funct(x_a_s, angles_a_s, label_points=False)
        for j, b in enumerate(x_b):
            x_b_s, angles_b_s = shift_polygon(x_b, angles_b, j)
            # tf.plot_discrete_funct(x_b_s, angles_b_s, label_points=False)
            # print x_a_s, angles_a_s, x_b_s, angles_b_s
            min_aux = subtract_histograms(x_a_s, angles_a_s, x_b_s, angles_b_s)
            # print min_aux
            if minimum_difference > min_aux:
                minimum_difference = min_aux
    return minimum_difference


if __name__ == '__main__':
    # turning_function(poly_1)
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
    # x_1, angles_1 = tf.turning_function(poly_1, plot_func=False)
    # tf.plot_polygon(poly_1)
    poly_b = np.array([
        [0, 0],
        [0, 2],
        [6, 2],
        [6, 0],
        [5, 0],
        [5, 1],
        [4, 1],
        [4, 0],
    ])
    poly_c = np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ])
    poly_d = np.array([
        [0, 0],
        [-1, 1 / 2.],
        [0, 1],
        [-1 / 2., 1 / 4.],
        [1, 0]
    ])
    poly_e = np.array([
        [0, 0],
        [-1, 0],
        [0, 1],
        [-1 / 2., 1 / 4.],
        [1, 0]
    ])
    poly_f = np.array([
        [0, 0],
        [0, 2],
        [3, 2],
        [3, 1.25],
        [4, 1.25],
        [4, 2],
        [6, 2],
        [6, 0],
        [4, 0],
        [4, 1],
        [3, 1],
        [3, 0],
    ])


    # x_2, angles_2 = tf.turning_function(poly_2, plot_func=False)
    # tf.plot_polygon(poly_2)
    poly_2 = mp.scale_polygon(poly_1, 10.3)
    poly_3 = mp.rotate_polygon(poly_1, np.pi / 3.)
    poly_4 = mp.translate_polygon(poly_1, 2.3, -1.01)
    poly_5 = mp.distort_polygon(poly_1, 0.1)
    tf.plot_polygon(poly_1)
    tf.plot_polygon(poly_2)
    tf.plot_polygon(poly_3)
    tf.plot_polygon(poly_4)
    tf.plot_polygon(poly_5)
    tf.plot_polygon(poly_b)
    tf.plot_polygon(poly_c)
    tf.plot_polygon(poly_d)
    tf.plot_polygon(poly_e)
    tf.plot_polygon(poly_f)
    print polygon_comparison(poly_1, poly_1)
    print polygon_comparison(poly_1, poly_2)
    print polygon_comparison(poly_1, poly_3)
    print polygon_comparison(poly_1, poly_4)
    print polygon_comparison(poly_1, poly_5)
    print polygon_comparison(poly_1, poly_b)
    print polygon_comparison(poly_1, poly_c)
    print polygon_comparison(poly_1, poly_d)
    print polygon_comparison(poly_1, poly_e)
    print polygon_comparison(poly_1, poly_f)