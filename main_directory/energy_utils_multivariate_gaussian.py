import numpy as np
import utils
import energy_utils_mean as mean_utils
import ctypes_utils as ctypes
import opencv_utils as cv_ut
from MaskClass import MaskClass
import mixture_gaussian

'''
                        MIXTURE GAUSSIAN ENERGY
'''


def multivariate_initialize_seed(CAC, from_gt=True):
    image = CAC.image_obj.image
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

    inside_gmm = get_values_in_region(inside_coordinates, image)
    outside_gmm = get_values_in_region(outside_coordinates, image)
    return inside_gmm, outside_gmm


def get_values_in_region(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    values_in_region = image[omega_coord_aux[:, 0], omega_coord_aux[:, 1]]
    gmm = mixture_gaussian.get_mixture_gaussian(values_in_region, 1, 'full')
    return gmm

def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image):
    omega_mean, omega_std = gmm.means_, gmm.covars_
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    omega_std = omega_std[0]
    k = len(omega_std)
    term_1 = len(aux) * k * np.log(2 * np.pi)
    term_2 = len(aux) * np.log(np.linalg.det(omega_std))
    sigma = np.linalg.inv(omega_std)
    x_sigma = np.dot(aux, sigma)
    term_3 = np.multiply(x_sigma, aux)
    term_3 = np.sum(term_3, axis=1)
    region_energy = term_1 + term_2 + sum(term_3)
    return region_energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient, weight=[1, 1, 1]):
    # E_mean

    image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0], 0),
                                        utils.evaluate_image(omega_coord, image_gradient[1], 0)])
    omega_mean, omega_std = gmm.means_, gmm.covars_
    omega_std = omega_std[0]
    sigma = np.linalg.inv(omega_std)
    aux = utils.evaluate_image(omega_coord, image, omega_mean)
    aux -= omega_mean
    sigma_aux = np.dot(sigma, aux.T)
    grad = gradient_gauss_energy_for_each_vertex(sigma_aux, affine_omega_coord, image_gradient_by_point)
    grad = np.transpose(grad, (2, 0, 1))
    return grad


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    aux = np.tile(aux, (2, 1, 1))
    first_prod = np.multiply(aux, np.transpose(image_gradient_by_point, (0, 2, 1)))
    second_prod = np.dot(first_prod, affine_omega_coord)
    return second_prod