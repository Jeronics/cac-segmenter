__author__ = 'jeroni'
from main_directory import utils
from ctypes_utils import *
import numpy as np


'''

                        LEARNING RATE

'''


def first_step_alpha(beta, curr_cage, grad_k):
    step = 0.001
    alpha = 0.05
    while all(multiple_norm(alpha * grad_k) <= beta):
        alpha += step
    return alpha


def second_step_alpha(alpha, curr_cage, grad_k, band_size, affine_contour_coord, contour_size, current_energy, image):
    step = 0.1
    print current_energy, type(current_energy)
    next_energy = current_energy+1
    alpha=alpha+step
    nrow, ncol = image.shape
    while current_energy < next_energy:
        alpha-=step

        # calculate new contour_coord
        contour_coord = np.dot(affine_contour_coord, curr_cage - grad_k*alpha)

        # Calculate new omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
        omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = get_omega_1_and_2_coord(band_size, contour_coord,
                                                                                           contour_size, ncol, nrow)

        affine_omega_1_coord, affine_omega_2_coord = get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                    omega_2_coord, omega_2_size,
                                                                                    len(curr_cage), curr_cage - grad_k*alpha)

        next_energy = mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image)
        print 'A, EN: ', alpha, next_energy


'''

                        MEAN ENERGY

'''


def mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    omega_1 = mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
    omega_2 = mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
    energy = (omega_1 + omega_2) / float(2)
    return energy


def mean_energy_grad(omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    # Calculate Image gradient
    image_gradient = np.array(np.gradient(image))

    # Calculate Energy:
    omega_1 = mean_energy_grad_per_region(omega1_coord, affine_omega_1_coord, image, image_gradient)
    omega_2 = mean_energy_grad_per_region(omega2_coord, affine_omega_2_coord, image, image_gradient)
    energy = omega_1 + omega_2
    return energy


def mean_energy_per_region(omega_coord, affine_omega_coord, image):
    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    return np.dot(aux, np.transpose(aux))


def mean_energy_grad_per_region(omega_coord, affine_omega_coord, image, image_gradient):
    # E_mean
    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    image_gradient_by_point = [utils.evaluate_image(omega_coord, image_gradient[0], 0),
                               utils.evaluate_image(omega_coord, image_gradient[1], 0)]
    mean_derivative = np.dot(image_gradient_by_point, affine_omega_coord) / float(len(omega_coord))
    grad = gradient_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point, mean_derivative)
    return grad  # *(1/pow(omega_std, 2)) for GAUSS


def gradient_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point, mean_derivative):
    # image_gradient_by_point = np.transpose(image_gradient_by_point)
    aux_x = np.multiply(np.transpose(affine_omega_coord), image_gradient_by_point[0])
    aux_y = np.multiply(np.transpose(affine_omega_coord), image_gradient_by_point[1])

    aux_x = np.dot(aux, np.transpose(aux_x) - mean_derivative[0])
    aux_y = np.dot(aux, np.transpose(aux_y) - mean_derivative[1])
    aux_1 = np.transpose([aux_x, aux_y])
    return aux_1


def get_omega_mean(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    omega_intensity = sum(image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]])
    omega_mean = omega_intensity / (len(omega_boolean[omega_boolean]))
    omega_std = np.std(image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]])
    return omega_mean, omega_std


'''

                    STOP CRITERIA

'''


def cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
    '''
    Checks if vertices cannot evolve anymore
    :param grad_k_3:
    :param grad_k_2:
    :param grad_k_1:
    :param grad_k:
    :return:
    '''
    if not all(np.diagonal(np.dot(grad_k, np.transpose(grad_k_2))) > 0):
        return False
    if not all(np.diagonal(np.dot(grad_k_1, np.transpose(grad_k_3))) > 0):
        return False
    if not all(np.diagonal(np.dot(grad_k, np.transpose(grad_k_1))) > 0):
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
