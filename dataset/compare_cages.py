__author__ = 'jeroni'
import turning_function as tf
import numpy as np
import collections



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
    lengths = x - x_
    integral = sum([a * b for a, b in zip(lengths, y)])
    return integral


if __name__ == '__main__':
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
    x_1, angles_1 = tf.turning_function(poly_1, plot_func=False)
    # tf.plot_polygon(poly_1)
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
    # tf.plot_polygon(poly_2)

    angles_diff = angles_2 - angles_1, 2
    angles_diff_sq = np.power(angles_diff)
    integral = integral_of_discrte_function([1 / 3., 2 / 3., 1], [1, 1, 1], interval=[0, 1])
    print integral