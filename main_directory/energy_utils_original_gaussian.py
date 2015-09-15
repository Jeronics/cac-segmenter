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


def get_values_in_region(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    values_in_region = image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]]
    values_in_region = np.array([values_in_region]).T
    omega_mean = np.mean(values_in_region)
    omega_std = np.std(values_in_region)
    return omega_mean, omega_std


def gauss_energy_per_region(omega_coord, affine_omega_coord, image):
    x_ = utils.evaluate_image(omega_coord, image)
    omega_mean = np.mean(x_)
    omega_std = np.std(x_)

    aux = x_ - omega_mean
    a = len(aux) * np.log(omega_std)
    b = 1 / float(2 * np.power(omega_std, 2))
    region_energy = a + np.dot(aux, np.transpose(aux)) * b
    return region_energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, image, image_gradient):
    x_ = utils.evaluate_image(omega_coord, image)
    omega_mean = np.mean(x_)
    omega_std = np.std(x_)
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