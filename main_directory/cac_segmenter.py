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


def cac_segmenter(image_obj, mask_obj, cage_obj, curr_cage_file, plot_evolution=False):
    start = time.time()
    image = image_obj.gray_image
    if cage_out_of_the_picture(cage_obj.cage, image_obj.shape):
        print 'Cage is out of the image! Not converged. Try a smaller cage'
        return None
    contour_coord, contour_size = get_contour(mask_obj)
    affine_contour_coordinates = get_affine_contour_coordinates(contour_coord, cage_obj.cage)

    # Update Step of contour coordinates
    contour_coord = np.dot(affine_contour_coordinates, cage_obj.cage)

    # copy of cage_obj
    iter = 0
    max_iter = 50
    max_iter_step_2 = 10
    first_stage = True
    grad_k_3, grad_k_2, grad_k_1, grad_k = np.zeros([cage_obj.num_points, 2]), np.zeros(
        [cage_obj.num_points, 2]), np.zeros(
        [cage_obj.num_points, 2]), np.zeros([cage_obj.num_points, 2])
    mid_point = sum(cage_obj.cage, 0) / cage_obj.num_points

    # PARAMETERS #
    # pixel steps
    beta = 5

    # Omega1 band size
    band_size = 500

    # Constraint Energy parameters
    # constraint energy. k=0 is none.
    k = 2

    # Algorithm requires k>=2*beta to work.
    d = 2 * beta
    constraint_params = [d, k]
    continue_while = True
    while continue_while:
        if iter > max_iter:
            continue_while = False
            print 'Maximum iterations reached'

        if cage_out_of_the_picture(cage_obj.cage, image_obj.shape):
            print 'Cage is out of the image! Not converged. Try a smaller cage'
            return None
        omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = get_omega_1_and_2_coord(band_size, contour_coord,
                                                                                           contour_size, mask_obj.width,
                                                                                           mask_obj.height)

        affine_omega_1_coord, affine_omega_2_coord = get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                    omega_2_coord, omega_2_size,
                                                                                    cage_obj.num_points, cage_obj.cage)

        # Update gradients
        grad_k_3 = grad_k_2.copy()
        grad_k_2 = grad_k_1.copy()
        grad_k_1 = grad_k.copy()
        grad_k = energies.mean_energy_grad(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                           image) + energies.grad_energy_constraint(cage_obj.cage, d, k)
        print 'Before:'
        print grad_k[0:3]
        grad_k = energies.multiple_normalize(grad_k)
        print 'After: '
        print grad_k[0:3]
        if first_stage:
            mid_point = sum(cage_obj.cage, 0) / float(cage_obj.num_points)
            axis = mid_point - cage_obj.cage
            axis = energies.multiple_normalize(axis)
            grad_k = energies.multiple_project_gradient_on_axis(grad_k, axis)
            alpha = beta

        else:
            energy = energies.mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                          image) + energies.energy_constraint(cage_obj.cage, d, k)
            alpha_new = energies.second_step_alpha(alpha, cage_obj.cage, grad_k, band_size, affine_contour_coordinates,
                                                   contour_size, energy, image, constraint_params)
            if alpha_new == 0:
                continue_while = False
                print 'Local minimum reached. no better alpha'
                # return curr_cage_file

        # Calculate alpha
        # grad_k = normalize_vectors(grad_k)
        # print grad_k[2], curr_cage_file[2]
        # print curr_cage_file[-2], grad_k[-2]
        alpha = beta  # find_optimal_alpha(beta, curr_cage_file, grad_k)

        # if iter % 20 == 0:
        # plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=[0., 0., 255.],
        # points2=cage_obj.cage - alpha * 10 * grad_k)

        if plot_evolution:
            plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=[0., 0., 255.],
                               points2=cage_obj.cage - alpha * 10 * grad_k)

        # Update File current cage
        cage_obj.cage += - alpha * grad_k
        if first_stage and energies.cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
            first_stage = False
            print 'First stage reached'

        # Update contour coordinates
        contour_coord = np.dot(affine_contour_coordinates, cage_obj.cage)
        iter += 1

    return cage_obj

    # TODO
    # IMPLEMENTAR CaLCUL DE ENERGIA I GRADIENT N-Dimensional


if __name__ == '__main__':
    # 1 ../test/ovella/image_ovella.png ../test/ovella/mask_01.png ../test/ovella/cage_01.txt
    # TODO: RUN 1 ../dataset/pear/pear2/pear2.png   ../dataset/pear/pear2/mask_00.png  ../dataset/pear/pear2/cage_8_1.5.txt
    #
    # # rgb_image, mask_file, init_cage_file, curr_cage_file = get_inputs(sys.argv)


    # image_obj = ImageClass()
    # image_obj.read_png('../dataset/pear/pear2/pear2.png')
    # mask_obj = MaskClass()
    # mask_obj.read_png('../dataset/pear/pear2/mask_00.png')
    # cage_obj = CageClass()
    # cage_obj.read_txt('../dataset/pear/pear2/cage_16_1.05.txt')
    # curr_cage_file = None
    # resulting_cage = cac_segmenter(image_obj, mask_obj, cage_obj, curr_cage_file)

    # image_obj = ImageClass()
    # image_obj.read_png('../dataset/eagle/eagle3/eagle3.png')
    # mask_obj = MaskClass()
    # mask_obj.read_png('../dataset/eagle/eagle3/mask_00.png')
    # cage_obj = CageClass()
    # cage_obj.read_txt('../dataset/eagle/eagle3/cage_16_1.05.txt')
    # curr_cage_file = None
    # resulting_cage = cac_segmenter(image_obj, mask_obj, cage_obj, curr_cage_file, plot_evolution=True)

    # image_obj = ImageClass()
    # image_obj.read_png('../dataset/apple/apple4/apple4.png')
    # mask_obj = MaskClass()
    # mask_obj.read_png('../dataset/apple/apple4/mask_00.png')
    # cage_obj = CageClass()
    # cage_obj.read_txt('../dataset/apple/apple4/cage_16_1.05.txt')
    # curr_cage_file = None
    # resulting_cage = cac_segmenter(image_obj, mask_obj, cage_obj, curr_cage_file, plot_evolution=True)


    image_obj = ImageClass()
    image_obj.read_png('../dataset/banana/banana2/banana2.png')
    mask_obj = MaskClass()
    mask_obj.read_png('../dataset/banana/banana2/mask_00.png')
    cage_obj = CageClass()
    cage_obj.read_txt('../dataset/banana/banana2/cage_16_1.05.txt')
    curr_cage_file = None
    resulting_cage = cac_segmenter(image_obj, mask_obj, cage_obj, curr_cage_file)