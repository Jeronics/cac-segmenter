__author__ = 'jeroni'

from ctypes_utils import *
# from ctypes import *
import time
import sys

import numpy as np

from utils import *
import energyutils
import energies

from scipy import interpolate


def cac_segmenter(image_obj, mask_obj, cage_obj, curr_cage_file):
    start = time.time()
    image = image_obj.gray_image

    num_control_point = cage_obj.num_points

    contour_coord, contour_size = get_contour(mask_obj)

    affine_contour_coordinates = get_affine_contour_coordinates(contour_coord, cage_obj.cage)

    # Update Step of contour coordinates
    contour_coord = np.dot(affine_contour_coordinates, cage_obj.cage)

    # copy of cage_obj
    iter = 0
    max_iter = 100
    first_stage = True
    grad_k_3, grad_k_2, grad_k_1, grad_k = np.zeros([num_control_point, 2]), np.zeros([num_control_point, 2]), np.zeros(
        [num_control_point, 2]), np.zeros([num_control_point, 2])
    mid_point = sum(cage_obj.cage, 0) / cage_obj.num_points
    beta = 20
    while iter < max_iter:

        band_size = 200
        omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = get_omega_1_and_2_coord(band_size, contour_coord,
                                                                                           contour_size, mask_obj.width, mask_obj.height)

        affine_omega_1_coord, affine_omega_2_coord = get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                    omega_2_coord, omega_2_size,
                                                                                    num_control_point, cage_obj.cage)

        if first_stage:
            # multiple_norm()
            # Update gradients
            grad_k_3 = grad_k_2.copy()
            grad_k_2 = grad_k_1.copy()
            grad_k_1 = grad_k.copy()
            grad_k = energies.mean_energy_grad(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                               image)

            # mid_point = sum(curr_cage_file, 0) / curr_cage_file.shape[0]
            # axis = mid_point - curr_cage_file
            # axis = normalize_vectors(axis)
            # grad_k = multiple_project_gradient_on_axis(grad_k, axis)
            grad_k = energies.multiple_normalize(grad_k)
        else:
            energy = energies.mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                          image)
            energies.second_step_alpha(alpha, cage_obj.cage, grad_k, band_size, affine_contour_coordinates,
                                       contour_size, energy,
                                       image)
            # return curr_cage_file

        # Calculate alpha
        # grad_k = normalize_vectors(grad_k)
        # print grad_k[2], curr_cage_file[2]
        # print curr_cage_file[-2], grad_k[-2]
        alpha = beta  # find_optimal_alpha(beta, curr_cage_file, grad_k)

        if iter % 20 == 0:
            plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=[0., 0., 255.],
                               points2=cage_obj.cage - alpha * 10 * grad_k)
        # plotContourOnImage(contour_coordinates, rgb_image, points=curr_cage_file, color=[0., 0., 255.],
        # points2=curr_cage_file - alpha * 10 * grad_k)

        # Update File current cage
        cage_obj.cage = cage_obj.cage - alpha * grad_k
        if first_stage and energies.cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
            first_stage = False

        # Update contour coordinates
        contour_coord = np.dot(affine_contour_coordinates, cage_obj.cage)
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

    # rgb_image, mask_file, init_cage_file, curr_cage_file = get_inputs(sys.argv)
    image_obj = ImageClass()
    image_obj.read_png('../dataset/pear/pear2/pear2.png')
    mask_obj = MaskClass()
    mask_obj.read_png('../dataset/pear/pear2/mask_00.png')
    cage_obj = CageClass()
    cage_obj.read_txt('../dataset/pear/pear2/cage_8_1.5.txt')
    curr_cage_file = None
    print cac_segmenter(image_obj, mask_obj, cage_obj, curr_cage_file)