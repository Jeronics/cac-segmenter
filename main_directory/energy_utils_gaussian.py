import numpy as np
import utils
import energy_utils_mean as mean_utils
import ctypes_utils as ctypes
import opencv_utils as cv_ut
from MaskClass import MaskClass

'''
                        GAUSSIAN ENERGY
'''


def initialize_seed(CAC, band_size):
    # Calculate Image gradient
    image = CAC.image_obj.gray_image
    center = CAC.mask_obj.center
    radius_point = CAC.mask_obj.radius_point
    print 'CENTER:', center
    print 'RADIUS POINT:', radius_point
    print 'RADIUS:', np.linalg.norm(np.array(radius_point) - np.array(center))
    radius = np.linalg.norm(np.array(radius_point) - np.array(center))

    inside_seed_omega = [center[0] + radius * 0.2, center[1]]
    outside_seed_omega = [center[0] + radius * 1.5, center[1]]
    inside_mask_seed = MaskClass()
    outside_mask_seed = MaskClass()
    inside_mask_seed.from_points_and_image(center, inside_seed_omega,image)
    outside_mask_seed.from_points_and_image(center, outside_seed_omega, image)
    inside_mask_seed.plot_image()
    CAC.mask_obj.plot_image()
    outside_mask_seed.plot_image()
    mask = CAC.mask_obj.mask / 255.
    mask = np.array(mask, dtype=np.float32)
    m = cv_ut.np_to_array(mask)
    CAC.mask_obj.plot_image()
    contour_coord, contour_size = ctypes.get_contour(CAC.mask_obj)
    omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = ctypes.get_omega_1_and_2_coord(band_size, contour_coord,
                                                                                              contour_size,
                                                                                              CAC.mask_obj.width,
                                                                                              CAC.mask_obj.height)
    omega_1_mean, omega_1_std = mean_utils.get_omega_mean(omega_1_coord, image)
    omega_2_mean, omega_2_std = mean_utils.get_omega_mean(omega_2_coord, image)
    return omega_1_mean, omega_1_std, omega_2_mean, omega_2_std


def gauss_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    '''
    Computes the Gaussian Energy of an Image
    :param omega_1_coord (numpy array): Omega coordinates for region Omega 1
    :param omega_2_coord (numpy array): Omega coordinates for region Omega 2
    :param affine_omega_1_coord (numpy array): Affine coordinates for region Omega 1
    :param affine_omega_2_coord (numpy array): Affine coordinates for region Omega 2
    :param image (numpy array): The Image
    :return:
    '''
    omega_1 = gauss_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
    omega_2 = gauss_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
    energy = (omega_1 + omega_2) / float(2)
    return energy


def grad_gauss_energy(omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    '''
    Computes the derivative of the Gaussian Energy of an Image with respect to the control points
    :param omega1_coord (numpy array): Omega coordinates for region Omega 1
    :param omega2_coord (numpy array): Omega coordinates for region Omega 2
    :param affine_omega_1_coord (numpy array): Affine coordinates for region Omega 1
    :param affine_omega_2_coord (numpy array): Affine coordinates for region Omega 2
    :param image (numpy array): The Image
    :return:
    '''
    # Calculate Image gradient
    image_gradient = np.array(np.gradient(image))

    # Calculate Energy Per region:
    omega_1 = grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, image, image_gradient)
    omega_2 = grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, image, image_gradient)

    energy = omega_1 + omega_2
    return energy


def gauss_energy_per_region(omega_coord, affine_omega_coord, image):
    omega_mean, omega_std = mean_utils.get_omega_mean(omega_coord, image)

    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    a = len(aux) * np.log(omega_std)
    b = 1 / (2 * np.power(omega_std))
    region_energy = a + np.dot(aux, np.transpose(aux)) * b
    return region_energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, image, image_gradient):
    # E_mean
    omega_mean, omega_std = mean_utils.get_omega_mean(omega_coord, image)
    print omega_mean, omega_std
    b = 1 / (np.power(omega_std, 2))
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    image_gradient_by_point = b * np.array([utils.evaluate_image(omega_coord, image_gradient[0], 0),
                                            utils.evaluate_image(omega_coord, image_gradient[1], 0)])
    grad = gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point)
    return grad  # *(1/pow(omega_std, 2)) for GAUSS


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    # image_gradient_by_point = np.transpose(image_gradient_by_point)
    first_prod = np.multiply(aux, affine_omega_coord.T)
    second_prod = np.dot(first_prod, image_gradient_by_point.T)
    return second_prod