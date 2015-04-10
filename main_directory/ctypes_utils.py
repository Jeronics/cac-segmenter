__author__ = 'jeroni'
from ctypes import *
import numpy as np
import utils

import os
libcac = CDLL("apicac/libcac.so")

cac_contour_get_interior_contour = libcac.cac_contour_get_interior_contour
cac_get_affine_coordinates = libcac.cac_get_affine_coordinates
cac_get_omega1_omega2 = libcac.cac_get_omega1_omega2

LP_c_int = POINTER(c_int)
LP_c_double = POINTER(c_double)
LP_LP_c_double = POINTER(LP_c_double)


def get_contour(mask_obj):
    mask = mask_obj.mask
    nrow, ncol = mask.shape
    contour_size = c_int()
    mat = LP_c_double()
    cac_contour_get_interior_contour(byref(contour_size), byref(mat), np.ctypeslib.as_ctypes(mask), c_int(ncol),
                                     c_int(nrow), c_int(4))
    contour_coord = np.ctypeslib.as_array(mat, shape=(contour_size.value, 2))
    return contour_coord, contour_size


def get_affine_contour_coordinates(contour_coord, init_cage_file):
    num_control_point = init_cage_file.shape[0]
    affine_contour_coordinates = np.zeros([contour_coord.shape[0], num_control_point])

    # Calculate the affine contour coordinates
    cac_get_affine_coordinates(np.ctypeslib.as_ctypes(affine_contour_coordinates), c_int(contour_coord.shape[0]),
                               np.ctypeslib.as_ctypes(contour_coord), c_int(num_control_point),
                               np.ctypeslib.as_ctypes(init_cage_file))
    return affine_contour_coordinates


def get_omega_1_and_2_coord(band_size, contour_coord, contour_size, ncol, nrow):
    # Allocate memory
    omega_1_size = c_int()
    omega_1_coord = LP_c_double()
    omega_2_size = c_int()
    omega_2_coord = LP_c_double()

    # Get contour OMEGA 1 and OMEGA 2
    cac_get_omega1_omega2(byref(omega_1_size), byref(omega_1_coord), byref(omega_2_size), byref(omega_2_coord),
                          contour_size, np.ctypeslib.as_ctypes(contour_coord), c_int(ncol), c_int(nrow),
                          c_int(band_size))

    omega_1_size = np.ctypeslib.as_array(omega_1_size)
    omega_2_size = np.ctypeslib.as_array(omega_2_size)

    omega_1_coord = np.ctypeslib.as_array(omega_1_coord, shape=(omega_1_size, 2))
    omega_2_coord = np.ctypeslib.as_array(omega_2_coord, shape=(omega_2_size, 2))
    return omega_1_coord, omega_2_coord, omega_1_size, omega_2_size


def get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size, omega_2_coord, omega_2_size, num_control_point,
                                   init_cage_file):
    # Get Affine coordinates OMEGA 1 and OMEGA 2. First Allocate affine_omega_i_coord:
    affine_omega_1_coord = np.zeros([omega_1_size, num_control_point])
    affine_omega_2_coord = np.zeros([omega_2_size, num_control_point])

    cac_get_affine_coordinates(np.ctypeslib.as_ctypes(affine_omega_1_coord), c_int(omega_1_size),
                               np.ctypeslib.as_ctypes(omega_1_coord), c_int(num_control_point),
                               np.ctypeslib.as_ctypes(init_cage_file))
    cac_get_affine_coordinates(np.ctypeslib.as_ctypes(affine_omega_2_coord), c_int(omega_2_size),
                               np.ctypeslib.as_ctypes(omega_2_coord), c_int(num_control_point),
                               np.ctypeslib.as_ctypes(init_cage_file))
    return affine_omega_1_coord, affine_omega_2_coord


mask_obj = utils.MaskClass()
mask_obj.read_png('../dataset/apple/apple5/mask_00.png')

coords, size = get_contour(mask_obj)

print coords

