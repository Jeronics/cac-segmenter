__author__ = 'jeroni'

from ctypes import *
import time

import numpy as np

from utils import *
from main_directory.energyutils import *
from scipy import interpolate
# 1 ../test/ovella/image_ovella.png ../test/ovella/mask_01.png ../test/ovella/cage_01.txt
# TODO: RUN 1 ../dataset/pear/pear2/pear2.png   ../dataset/pear/pear2/mask_00.png  ../dataset/pear/pear2/cage_8_1.3.txt
def cac_segmenter(rgb_image, mask_file, init_cage_file, curr_cage_file):
    start = time.time()
    # Read inputs, return numpy arrays for image, cage/s and mask.

    print rgb_image.shape
    printNpArray(rgb_image)
    # TURN IMAGE TO GRAYSCALE!
    image = rgb2gray(rgb_image)
    printNpArray(image)

    # testlibrary = CDLL("testlibrary.so")
    libcac = CDLL("apicac/libcac.so")

    cac_contour_get_interior_contour = libcac.cac_contour_get_interior_contour
    cac_get_affine_coordinates = libcac.cac_get_affine_coordinates
    cac_get_omega1_omega2 = libcac.cac_get_omega1_omega2

    LP_c_int = POINTER(c_int)  # mateixa notacio que python
    LP_c_double = POINTER(c_double)  # mateixa notacio que python
    LP_LP_c_double = POINTER(LP_c_double)  # mateixa notacio que python

    # millor no posar-ho per "saltar-nos" un error que indica el python. No se com arreglar-lo
    # cac_contour_get_interior_contour.argtypes=[LP_c_int, LP_LP_c_double, LP_c_double, c_int, c_int, c_int]

    nrow, ncol = mask_file.shape
    size_image = image.shape
    # img = np.copy(mask_file)

    contour_size = c_int()  # un sencer
    mat = LP_c_double()  # un punter a double

    # Amb 'byref' passem la referencia a la variable
    # Amb as_types transformem de tipus numpy a tipus ctypes, cosa que va millor
    cac_contour_get_interior_contour(byref(contour_size), byref(mat), ctypeslib.as_ctypes(mask_file), c_int(ncol),
                                     c_int(nrow), c_int(4))

    # passem la matriu retornada a tipus numpy. Observa com defineixo la mida de la matriu
    contour_coordinates = ctypeslib.as_array(mat, shape=(contour_size.value, 2))

    # OPTIONAL: PRINT THE CONTOUR ON THE IMAGE
    num_control_point = init_cage_file.shape[0]
    affine_contour_coordinates = np.zeros([contour_coordinates.shape[0], num_control_point])

    # Calculate the affine contour coordinates
    cac_get_affine_coordinates(ctypeslib.as_ctypes(affine_contour_coordinates), c_int(contour_coordinates.shape[0]),
                               ctypeslib.as_ctypes(contour_coordinates), c_int(num_control_point),
                               ctypeslib.as_ctypes(init_cage_file))

    # Update Step of contour coordinates
    contour_coordinates = np.dot(affine_contour_coordinates, init_cage_file)

    curr_cage_file = init_cage_file
    iter = 0
    max_iter = 1000
    first_stage = True
    grad_k_3, grad_k_2, grad_k_1, grad_k = np.zeros([num_control_point, 2]), np.zeros([num_control_point, 2]), np.zeros(
        [num_control_point, 2]), np.zeros([num_control_point, 2])
    mid_point = sum(curr_cage_file, 0) / curr_cage_file.shape[0]
    beta = 5
    alpha = 10
    while (iter < max_iter):

        # Allocate memory
        omega1_size = c_int()
        omega1_coord = LP_c_double()
        omega2_size = c_int()
        omega2_coord = LP_c_double()
        band_size = 100

        # Get contour OMEGA 1 and OMEGA 2
        cac_get_omega1_omega2(byref(omega1_size), byref(omega1_coord), byref(omega2_size), byref(omega2_coord),
                              contour_size, ctypeslib.as_ctypes(contour_coordinates), c_int(ncol), c_int(nrow),
                              c_int(band_size))

        omega1_size = ctypeslib.as_array(omega1_size)
        omega2_size = ctypeslib.as_array(omega2_size)

        omega1_coord = ctypeslib.as_array(omega1_coord, shape=(omega1_size, 2))
        omega2_coord = ctypeslib.as_array(omega2_coord, shape=(omega2_size, 2))
        omega_common = []


        # Get Affine coordinates OMEGA 1 and OMEGA 2. First Allocate affine_omegai_coordinates:
        affine_omega1_coordinates = np.zeros([omega1_size, num_control_point])
        affine_omega2_coordinates = np.zeros([omega2_size, num_control_point])

        cac_get_affine_coordinates(ctypeslib.as_ctypes(affine_omega1_coordinates), c_int(omega1_size),
                                   ctypeslib.as_ctypes(omega1_coord), c_int(num_control_point),
                                   ctypeslib.as_ctypes(init_cage_file))
        cac_get_affine_coordinates(ctypeslib.as_ctypes(affine_omega2_coordinates), c_int(omega2_size),
                                   ctypeslib.as_ctypes(omega2_coord), c_int(num_control_point),
                                   ctypeslib.as_ctypes(init_cage_file))

        f_interior = file('interior_points.txt', 'w')
        f_exterior = file('exterior_points.txt', 'w')
        for coord in omega1_coord:
            f_interior.write("%d %d\n" % (coord[0], coord[1]))
        for coord in omega2_coord:
            f_exterior.write("%d %d\n" % (coord[0], coord[1]))

        f_interior.close()
        f_exterior.close()
        if not first_stage:
            print 'First Stage Complete'
            return curr_cage_file
            alpha = 1 * alpha
            first_stage = True
        if first_stage:
            # multiple_norm()
            # Update gradients
            grad_k_3 = grad_k_2.copy()
            grad_k_2 = grad_k_1.copy()
            grad_k_1 = grad_k.copy()
            grad_k = gradientEnergy(omega1_coord, omega2_coord, affine_omega1_coordinates, affine_omega2_coordinates,
                                    image)
            # mid_point = sum(curr_cage_file, 0) / curr_cage_file.shape[0]
            # axis = mid_point - curr_cage_file
            # axis = normalize_vectors(axis)
            # grad_k = multiple_project_gradient_on_axis(grad_k, axis)
            grad_k = multiple_normalize(grad_k)

        print 'Omega1: ', calculateOmegaMean(omega1_coord, image)
        print 'Omega2: ', calculateOmegaMean(omega2_coord, image), '\n'
        # Generate random movements
        # vertex_variations = np.random.random(init_cage_file.shape) * 3 - 1.

        # Calculate alpha
        # grad_k = normalize_vectors(grad_k)
        # print grad_k[2], curr_cage_file[2]
        # print curr_cage_file[-2], grad_k[-2]


        if iter % 20 == 0:
            plotContourOnImage(contour_coordinates, rgb_image, points=curr_cage_file, color=[0., 0., 255.],
                               points2=curr_cage_file - alpha * 10 * grad_k)

        curr_cage_file = curr_cage_file - alpha * grad_k
        if first_stage and cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
            first_stage = False

        # Update contour coordinates
        contour_coordinates = np.dot(affine_contour_coordinates, curr_cage_file)
        iter += 1


        # THE END
        # Time elapsed
        # end = time.time()
        # print end-start

        # TODO
        # IMPLEMENTAR CaLCUL DE ENERGIA I GRADIENT N-Dimensional
if __name__ == '__main__':
    rgb_image, mask_file, init_cage_file, curr_cage_file = get_inputs(sys.argv)
    cac_segmenter(rgb_image, mask_file, init_cage_file, curr_cage_file)