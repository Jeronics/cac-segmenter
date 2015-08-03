import numpy as np

import utils
import opencv_utils as opencv_ut
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

        inside_seed = inside_mask_seed.mask
        outside_seed = 255. - outside_mask_seed.mask

        inside_seed = opencv_ut.erode(inside_seed, width=10)
        outside_seed = opencv_ut.erode(outside_seed, width=10)
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

    print 'Number of components:'
    inside_gmm = get_values_in_region(inside_coordinates, image)
    print 'Interior:\t', inside_gmm.n_components
    outside_gmm = get_values_in_region(outside_coordinates, image)
    print 'Exterior:\t', outside_gmm.n_components
    return inside_gmm, outside_gmm


def get_values_in_region(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    values_in_region = image[omega_coord_aux[:, 0], omega_coord_aux[:, 1]]
    gmm = mixture_gaussian.get_number_of_components(values_in_region)
    return gmm


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
    omega_1 = gauss_energy_per_region(omega_1_coord, affine_omega_1_coord, self.inside_gmm, image)
    omega_2 = gauss_energy_per_region(omega_2_coord, affine_omega_2_coord, self.outside_gmm, image)
    energy = -(omega_1 + omega_2) / 2.
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
    omega_1 = grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, self.gmm, image, image_gradient)
    omega_2 = grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, self.gmm, image, image_gradient)

    energy = -(omega_1 + omega_2)
    return energy


def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image):
    region_energy = 0
    for i, (omega_mean, omega_std, omega_weight) in enumerate(zip(gmm.means_, gmm.covars_, gmm.weights_)):
        aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
        k = len(omega_std)
        term_1 = len(aux) * k * np.log(2 * np.pi)
        term_2 = len(aux) * np.log(np.linalg.det(omega_std))
        sigma = np.linalg.inv(omega_std)
        x_sigma = np.dot(aux, sigma)
        term_3 = np.multiply(x_sigma, aux)
        term_3 = np.sum(term_3, axis=1)
        region_energy += omega_weight * (term_1 + term_2 + sum(term_3))
    return region_energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient):
    grad = np.zeros([affine_omega_coord.shape[1], omega_coord.shape[1], image.shape[2]])
    image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0], 0),
                                        utils.evaluate_image(omega_coord, image_gradient[1], 0)])
    for i, (omega_mean, omega_std, omega_weight) in enumerate(zip(gmm.means_, gmm.covars_, gmm.weights_)):
        sigma = np.linalg.inv(omega_std)
        aux = utils.evaluate_image(omega_coord, image)
        aux -= omega_mean
        sigma_aux = np.dot(sigma, aux.T)
        grad_ = gradient_gauss_energy_for_each_vertex(sigma_aux, affine_omega_coord,
                                                      image_gradient_by_point) * omega_weight
        grad += np.transpose(grad_, (2, 0, 1))
    return grad


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point, ):
    # image_gradient_by_point = np.transpose(image_gradient_by_point)
    aux = np.tile(aux, (2, 1, 1))
    first_prod = np.multiply(aux, np.transpose(image_gradient_by_point, (0, 2, 1)))
    second_prod = np.dot(first_prod, affine_omega_coord)
    return second_prod