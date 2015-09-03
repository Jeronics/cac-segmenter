import numpy as np
import utils
import energy_utils_mean as mean_utils
import ctypes_utils as ctypes
import opencv_utils as cv_ut
from MaskClass import MaskClass
import mixture_gaussian

'''
                        GAUSSIAN ENERGY
'''


def initialize_seed(CAC, from_gt=True):
    image = CAC.image_obj.gray_image
    if from_gt:
        print 'Seed from ground truth...'
        inside_mask_seed = CAC.ground_truth_obj
        outside_mask_seed = CAC.ground_truth_obj
    else:
        center = CAC.mask_obj.center
        radius_point = CAC.mask_obj.radius_point
        print 'CENTER:', center
        print 'RADIUS POINT:', radius_point
        print 'RADIUS:', np.linalg.norm(np.array(radius_point) - np.array(center))
        radius = np.linalg.norm(np.array(radius_point) - np.array(center))

        inside_seed_omega = [center[0] + radius * 0.2, center[1]]
        outside_seed_omega = [center[0] + radius * 1.8, center[1]]

        inside_mask_seed = MaskClass()
        outside_mask_seed = MaskClass()

        inside_mask_seed.from_points_and_image(center, inside_seed_omega, image)
        outside_mask_seed.from_points_and_image(center, outside_seed_omega, image)

    inside_seed = inside_mask_seed.mask
    outside_seed = 255. - outside_mask_seed.mask
    # inside_mask_seed.plot_image()
    # CAC.mask_obj.plot_image()
    # utils.printNpArray(outside_seed)
    inside_coordinates = np.argwhere(inside_seed == 255.)
    outside_coordinates = np.argwhere(outside_seed == 255.)

    omega_in_mean, omega_in_std = get_values_in_region(inside_coordinates, image)
    omega_out_mean, omega_out_std = get_values_in_region(outside_coordinates, image)
    return omega_in_mean, omega_in_std, omega_out_mean, omega_out_std


def get_values_in_region(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    values_in_region = image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]]
    values_in_region = np.array([values_in_region]).T
    omega_mean = np.mean(values_in_region)
    omega_std = np.std(values_in_region)
    return omega_mean, omega_std


def gauss_energy_per_region(omega_coord, affine_omega_coord, omega_mean, omega_std, image):
    aux = utils.evaluate_image(omega_coord, image) - omega_mean
    a = len(aux) * np.log(omega_std)
    b = 1 / float(2 * np.power(omega_std, 2))
    region_energy = a + np.dot(aux, np.transpose(aux)) * b
    return region_energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, omega_mean, omega_std, image, image_gradient):
    b = 1 / (np.power(omega_std, 2))
    aux = utils.evaluate_image(omega_coord, image) - omega_mean
    image_gradient_by_point = b * np.array([utils.evaluate_image(omega_coord, image_gradient[0]),
                                            utils.evaluate_image(omega_coord, image_gradient[1])])
    grad = gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point)
    return grad


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    first_prod = np.multiply(aux, affine_omega_coord.T)
    second_prod = np.dot(first_prod, image_gradient_by_point.T)
    return second_prod