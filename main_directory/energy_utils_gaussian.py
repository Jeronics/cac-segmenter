import numpy as np
import utils

'''
                        GAUSSIAN ENERGY
'''


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
    omega_mean, omega_std = get_omega_mean(omega_coord, image)

    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    a = len(aux) * np.log(omega_std)
    b = 1 / (2 * np.pow(omega_std))
    region_energy = a + np.dot(aux, np.transpose(aux)) * b
    return region_energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, image, image_gradient):
    # E_mean
    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    b = 1 / (np.pow(omega_std, 2))
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    image_gradient_by_point = b * [utils.evaluate_image(omega_coord, image_gradient[0], 0),
                                   utils.evaluate_image(omega_coord, image_gradient[1], 0)]
    mean_derivative = np.dot(image_gradient_by_point, affine_omega_coord) / float(len(omega_coord))
    grad = gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point)
    return grad  # *(1/pow(omega_std, 2)) for GAUSS


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    # image_gradient_by_point = np.transpose(image_gradient_by_point)
    aux_x = np.multiply(np.transpose(affine_omega_coord), image_gradient_by_point[0])
    aux_y = np.multiply(np.transpose(affine_omega_coord), image_gradient_by_point[1])
    aux_x = np.dot(aux, np.transpose(aux_x))
    aux_y = np.dot(aux, np.transpose(aux_y))
    aux_1 = np.transpose([aux_x, aux_y])
    return aux_1