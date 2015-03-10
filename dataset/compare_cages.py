__author__ = 'jeroni'
import numpy as np
import math
from sympy import Polygon, Point
from ctypes import *
import numpy as np

# TARGET=libsim.so
#
# all:  $(TARGET)
# 	gcc -o sim sim.c -lm
# clean:
# 	rm sim
#
# all:
# 	gcc -o sim sim.c -lm
# clean:
# 	rm sim
#
# libcac = CDLL("pol_sym/libsim.so")
#
# read_poly = libcac.main
#



LP_c_int = POINTER(c_int)
LP_c_double = POINTER(c_double)
LP_LP_c_double = POINTER(LP_c_double)


p1 = Point(0, 0)
p2 = Point(1, 0)
p3 = Point(1, 1)
p4 = Point(0, 1)

x = Polygon(p1, p2, p3, p4)
print 'Area', x.area, x.perimeter
print x.centroid

print '\n'

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


agls = []

for p, a in x.angles.items():
    print 'angle', a
    agls.append(get_angle(a))
print '\n'
print agls
print sum(agls)

p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
poly = Polygon(p1, p2, p3, p4)
print poly.angles[p1]