__author__ = 'jeroni'

from ctypes import *
import time

import numpy as np

from utils import *
from main_directory.energyutils import *
from main_directory import energies as energy
from scipy import interpolate


def cac_segmenter(rgb_image, mask_file, init_cage_file, curr_cage_file):
    start = time.time()
    # Read inputs, return numpy arrays for image, cage/s and mask.

    print rgb_image.shape

    # TURN IMAGE TO GRAYSCALE!
    image = rgb2gray(rgb_image)

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
    contour_coord = ctypeslib.as_array(mat, shape=(contour_size.value, 2))

    # OPTIONAL: PRINT THE CONTOUR ON THE IMAGE
    num_control_point = init_cage_file.shape[0]
    affine_contour_coordinates = np.zeros([contour_coord.shape[0], num_control_point])

    # Calculate the affine contour coordinates
    cac_get_affine_coordinates(ctypeslib.as_ctypes(affine_contour_coordinates), c_int(contour_coord.shape[0]),
                               ctypeslib.as_ctypes(contour_coord), c_int(num_control_point),
                               ctypeslib.as_ctypes(init_cage_file))

    # Update Step of contour coordinates
    contour_coord = np.dot(affine_contour_coordinates, init_cage_file)

    curr_cage_file = init_cage_file.copy()
    iter = 0
    max_iter = 1000
    first_stage = True
    grad_k_3, grad_k_2, grad_k_1, grad_k = np.zeros([num_control_point, 2]), np.zeros([num_control_point, 2]), np.zeros(
        [num_control_point, 2]), np.zeros([num_control_point, 2])
    mid_point = sum(curr_cage_file, 0) / curr_cage_file.shape[0]
    beta = 5
    while (iter < max_iter):
        # Allocate memory
        omega_1_size = c_int()
        omega_1_coord = LP_c_double()
        omega_2_size = c_int()
        omega_2_coord = LP_c_double()
        band_size = 50

        # Get contour OMEGA 1 and OMEGA 2
        cac_get_omega1_omega2(byref(omega_1_size), byref(omega_1_coord), byref(omega_2_size), byref(omega_2_coord),
                              contour_size, ctypeslib.as_ctypes(contour_coord), c_int(ncol), c_int(nrow),
                              c_int(band_size))

        omega_1_size = ctypeslib.as_array(omega_1_size)
        omega_2_size = ctypeslib.as_array(omega_2_size)

        omega_1_coord = ctypeslib.as_array(omega_1_coord, shape=(omega_1_size, 2))
        omega_2_coord = ctypeslib.as_array(omega_2_coord, shape=(omega_2_size, 2))
        omega_common = []


        # Get Affine coordinates OMEGA 1 and OMEGA 2. First Allocate affine_omegai_coordinates:
        affine_omega_1_coord = np.zeros([omega_1_size, num_control_point])
        affine_omega_2_coord = np.zeros([omega_2_size, num_control_point])

        cac_get_affine_coordinates(ctypeslib.as_ctypes(affine_omega_1_coord), c_int(omega_1_size),
                                   ctypeslib.as_ctypes(omega_1_coord), c_int(num_control_point),
                                   ctypeslib.as_ctypes(init_cage_file))
        cac_get_affine_coordinates(ctypeslib.as_ctypes(affine_omega_2_coord), c_int(omega_2_size),
                                   ctypeslib.as_ctypes(omega_2_coord), c_int(num_control_point),
                                   ctypeslib.as_ctypes(init_cage_file))

        f_interior = file('interior_points.txt', 'w')
        f_exterior = file('exterior_points.txt', 'w')
        for coord in omega_1_coord:
            f_interior.write("%d %d\n" % (coord[0], coord[1]))
        for coord in omega_2_coord:
            f_exterior.write("%d %d\n" % (coord[0], coord[1]))

        f_interior.close()
        f_exterior.close()
        if not first_stage:
            print 'First Stage Complete'
            # return curr_cage_file

        if first_stage:
            # multiple_norm()
            # Update gradients
            grad_k_3 = grad_k_2.copy()
            grad_k_2 = grad_k_1.copy()
            grad_k_1 = grad_k.copy()
            grad_k = energy.mean_energy_grad(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                    image)

            # mid_point = sum(curr_cage_file, 0) / curr_cage_file.shape[0]
            # axis = mid_point - curr_cage_file
            # axis = normalize_vectors(axis)
            # grad_k = multiple_project_gradient_on_axis(grad_k, axis)
            print grad_k
            grad_k = multiple_normalize(grad_k)
            print grad_k
        else:
            energy.mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,image)
            print ''



        # print 'Omega1: ', calculateOmegaMean(omega1_coord, image)
        # print 'Omega2: ', calculateOmegaMean(omega2_coord, image), '\n'
        # Generate random movements
        # vertex_variations = np.random.random(init_cage_file.shape) * 3 - 1.

        # Calculate alpha
        # grad_k = normalize_vectors(grad_k)
        # print grad_k[2], curr_cage_file[2]
        # print curr_cage_file[-2], grad_k[-2]
        alpha = beta # find_optimal_alpha(beta, curr_cage_file, grad_k)
        print 'A, B:', alpha, beta


        # if iter % 20 == 0:
        # plotContourOnImage(contour_coordinates, rgb_image, points=curr_cage_file, color=[0., 0., 255.],
        #                        points2=curr_cage_file - alpha * 10 * grad_k)

        # Update File current cage
        curr_cage_file = curr_cage_file - alpha * grad_k
        if first_stage and cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
            first_stage = False

        # Update contour coordinates
        contour_coord = np.dot(affine_contour_coordinates, curr_cage_file)
        iter += 1

    return None
    # THE END
    # Time elapsed
    # end = time.time()
    # print end-start

    # TODO
    # IMPLEMENTAR CaLCUL DE ENERGIA I GRADIENT N-Dimensional


if __name__ == '__main__':
    # 1 ../test/ovella/image_ovella.png ../test/ovella/mask_01.png ../test/ovella/cage_01.txt
    # TODO: RUN 1 ../dataset/pear/pear2/pear2.png   ../dataset/pear/pear2/mask_00.png  ../dataset/pear/pear2/cage_8_1.5.txt

    rgb_image, mask_file, init_cage_file, curr_cage_file = get_inputs(sys.argv)
    print cac_segmenter(rgb_image, mask_file, init_cage_file, curr_cage_file)