import sys

sys.path.append('/cac-segmenter')
import utils
import re
import numpy as np
from scipy import *
from matplotlib import pyplot
from scipy import ndimage
import scipy
from scipy import misc
import PIL
import math
from scipy import interpolate


def calculateOmegaMean(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    omega_intensity = sum(image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]])
    omega_mean = omega_intensity / (len(omega_boolean[omega_boolean == True]))
    omega_std = np.std(image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]])
    return omega_mean, omega_std


def calculateMeanEnergy(omega1_coord, omega2_coord, image):
    omega1Mean = calculateOmegaMean(omega1_coord, image)
    omega2Mean = calculateOmegaMean(omega2_coord, image)
    Energy1 = calcuateOmegaMeanEnergy(image, omega1Mean, omega1_coord)
    Energy2 = calcuateOmegaMeanEnergy(image, omega2Mean, omega2_coord)
    return (Energy1 + Energy2) / 2


def calcuateOmegaMeanEnergy(image, omegaMean, omega_coord):
    val = 0.
    for a in omega_coord:
        if (utils.is_inside_image(a, image.shape)):
            val += pow((image[a[0]][a[1]] - omegaMean), 2)
            # ELSE: DE MOMENT NO FER RES.
    return val


def calcuateOmegaMeanEnergyGradient(image_gradient, omegaMean, omega_coord):
    val = 0.
    cardinal = omega_coord
    for a in omega_coord:
        if (is_inside_image(a, image_gradient.shape)):
            val += pow((image_gradient[a[0]][a[1]] - omegaMean), 2)
        else:
            cardinal -= 1
    return val / cardinal


def energy_for_each_vertex():
    return 0


def gradient_energy_for_each_vertex(aux, affine_omega_coordinates, image_gradient_by_point, mean_derivative):
    # image_gradient_by_point = np.transpose(image_gradient_by_point)
    aux_x = np.multiply(np.transpose(affine_omega_coordinates), image_gradient_by_point[0])
    aux_y = np.multiply(np.transpose(affine_omega_coordinates), image_gradient_by_point[1])

    aux_x = np.dot(aux, np.transpose(aux_x) - mean_derivative[0])
    aux_y = np.dot(aux, np.transpose(aux_y) - mean_derivative[1])
    aux_1 = np.transpose([aux_x, aux_y])
    return aux_1


def GAUSS_gradient_energy_for_each_vertex(aux, affine_omega_coordinates, image_gradient_by_point):
    image_gradient_by_point_aux = np.transpose(image_gradient_by_point)
    aux_1 = np.multiply(aux, np.transpose(affine_omega_coordinates))
    aux_2 = np.dot(aux_1, image_gradient_by_point_aux)
    return aux_2


def gradientEnergy(omega1_coord, omega2_coord, affine_omega1_coordinates, affine_omega2_coordinates, image):
    # Calculate Image gradient
    image_gradient = np.array(np.gradient(image))
    # Calculate Energy:
    Omega1 = gradient_Energy_per_region(omega1_coord, affine_omega1_coordinates, image, image_gradient)
    Omega2 = gradient_Energy_per_region(omega2_coord, affine_omega2_coordinates, image, image_gradient)
    Energy = Omega1 + Omega2
    return Energy


def gradient_Energy_per_region(omega_coord, affine_omega_coordinates, image, image_gradient):
    # E_mean
    mean_omega, omega_std = calculateOmegaMean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, mean_omega) - mean_omega
    image_gradient_by_point = [utils.evaluate_image(omega_coord, image_gradient[0], 0),
                               utils.evaluate_image(omega_coord, image_gradient[1], 0)]
    mean_derivative = np.dot(image_gradient_by_point, affine_omega_coordinates) / float(len(omega_coord))
    grad = gradient_energy_for_each_vertex(aux, affine_omega_coordinates, image_gradient_by_point, mean_derivative)
    return grad  # *(1/pow(omega_std, 2)) for GAUSS


def cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
    '''
    Checks if vertices cannot evolve anymore
    :param grad_k_3:
    :param grad_k_2:
    :param grad_k_1:
    :param grad_k:
    :return:
    '''
    product_1 = sum(np.transpose(grad_k * grad_k_2), 0)
    product_2 = sum(np.transpose(grad_k_1 * grad_k_3), 0)
    product_3 = sum(np.transpose(grad_k * grad_k_1), 0)
    if any(x <= 0 for x in product_1):
        return False
    if any(x <= 0 for x in product_2):
        return False
    if any(x >= 0 for x in product_3):
        return False
    return True


def multiple_project_gradient_on_axis(a, b):
    '''
    Finds a's projection on b
    :param a:
    :param b:
    :return:
    '''
    return np.transpose((multiple_dot_products(a, b) / np.power(multiple_norm(b), 2)) * np.transpose(b))


def normalize_vectors(vectors):
    '''
    Normalizes vectors
    :param vect:
    :return:
    '''
    vectors_aux = np.array([x / np.linalg.norm(x) for x in vectors])
    return vectors_aux


'''
    THE FOLLOWING FUNCTIONS ARE MADE TO ACCEPT MULTIPLE VECTORS IN THE FOLLOWING FORMAT:

    v1
    v2
    .
    .
    .
    vN
'''


def multiple_norm(a):
    '''

    :param a:
    :return:
    '''
    return np.array([np.linalg.norm(x) for x in a])


def multiple_normalize(a):
    '''

    :param a:
    :return:
    '''
    return np.array([x / np.linalg.norm(x) for x in a])


def multiple_dot_products(a, b):
    c = a * b
    return np.array([sum(x) for x in c])
