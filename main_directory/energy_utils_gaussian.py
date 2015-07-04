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


def initialize_seed(CAC):
    # Calculate Image gradient
    image = CAC.image_obj.gray_image
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
    return omega_out_mean, omega_out_std, omega_in_mean, omega_in_std


def get_values_in_region(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    values_in_region = image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]]
    values_in_region = np.array([values_in_region]).T
    omega_mean = np.mean(values_in_region)
    omega_std = np.std(values_in_region)
    import pdb;

    pdb.set_trace()
    return omega_mean, omega_std


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
    energy = -(omega_1 + omega_2) / float(2)
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

    energy = -(omega_1 + omega_2)
    return energy


def gauss_energy_per_region(omega_coord, affine_omega_coord, omega_mean, omega_std, image):
    # omega_mean, omega_std = mean_utils.get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    a = len(aux) * np.log(omega_std)
    b = 1 / float(2 * np.power(omega_std, 2))
    region_energy = a + np.dot(aux, np.transpose(aux)) * b
    return region_energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, omega_mean, omega_std, image, image_gradient):
    # E_mean
    # omega_mean, omega_std = mean_utils.get_omega_mean(omega_coord, image)
    print 'Mean, Std:', omega_mean, omega_std
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