import numpy as np
from scipy.ndimage.filters import gaussian_filter

import utils
import opencv_utils as opencv_ut
from MaskClass import MaskClass
import mixture_gaussian


'''
                        MIXTURE GAUSSIAN ENERGY
'''


def multivariate_initialize_seed(CAC, from_gt=True):
    image = CAC.image_obj.image
    # image = gaussian_filter(image, sigma=0.5, order=0)

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


def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([w for w in gmm.weights_]).T

    x_ = utils.evaluate_image(omega_coord, image)
    x_m = np.transpose(np.tile(x_, (1, 1, 1)), (1, 0, 2)) - means
    # print x_m.shape
    denom = np.linalg.inv(covars)
    x_m_denom = np.dot(x_m, denom)
    x_m_denom = np.array([x_m_denom[:, i, i, :] for i in xrange(len(weights))])
    aux=-np.sum(np.multiply(np.transpose(x_m_denom, (1, 0, 2)), x_m), axis=2)/2.
    exp_x_m_denom_x_m = np.exp(aux)

    k = means.shape[1]
    coeff = 1 / (np.power(np.sqrt(2 * np.pi), k) * np.sqrt(np.linalg.det(covars)))
    mixt = exp_x_m_denom_x_m * coeff
    unsummed_mixture_prob = mixt * weights
    unsummed_mixture_prob = np.tile(unsummed_mixture_prob, (3, 1, 1))
    unsummed_mixture_prob = np.transpose(unsummed_mixture_prob, (1, 0, 2))

    if len(means) > 1:
        mixture_prob = np.array([np.sum(unsummed_mixture_prob, axis=2)])
    else:
        mixture_prob = np.transpose(unsummed_mixture_prob, (1, 0, 2)).T

    k = mixture_prob
    smallest_num = np.exp(-200)
    k[k < smallest_num] = smallest_num  # careful with when k[k>0] is empty
    mixture_prob = k

    energy= np.sum(np.log(mixture_prob))
    return energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([w for w in gmm.weights_]).T

    x_ = utils.evaluate_image(omega_coord, image)
    x_m = np.transpose(np.tile(x_, (1, 1, 1)), (1, 0, 2)) - means
    # print x_m.shape
    denom = np.linalg.inv(covars)
    x_m_denom = np.dot(x_m, denom)
    x_m_denom = np.array([x_m_denom[:, i, i, :] for i in xrange(len(weights))])
    aux=-np.sum(np.multiply(np.transpose(x_m_denom, (1, 0, 2)), x_m), axis=2)/2.
    exp_x_m_denom_x_m = np.exp(aux)

    k = means.shape[1]
    coeff = 1 / (np.power(np.sqrt(2 * np.pi), k) * np.sqrt(np.linalg.det(covars)))
    mixt = exp_x_m_denom_x_m * coeff
    unsummed_mixture_prob = mixt * weights
    unsummed_mixture_prob = np.tile(unsummed_mixture_prob, (3, 1, 1))
    unsummed_mixture_prob = np.transpose(unsummed_mixture_prob, (1, 0, 2))

    if len(means) > 1:
        mixture_prob = np.array([np.sum(unsummed_mixture_prob, axis=2)])
    else:
        mixture_prob = np.transpose(unsummed_mixture_prob, (1, 0, 2)).T

    k = mixture_prob
    smallest_num = np.exp(-200)
    k[k < smallest_num] = smallest_num  # careful with when k[k>0] is empty
    mixture_prob = k

    # caluculate 1/P(x)
    coeff_ = 1 / mixture_prob

    print 'energy', np.sum(np.log(mixture_prob))

    # calculate the derivative of the mixture
    denom_x_m = np.dot(denom, np.transpose(x_m, (0, 2, 1)))

    denom_x_m = np.array([denom_x_m[i, :, :, i] for i in xrange(len(weights))])

    unsummed_mixture_derivative = unsummed_mixture_prob * denom_x_m.T / 2.
    if len(means) > 1:
        mixture_derivative = np.array([np.sum(unsummed_mixture_derivative, axis=2)])
    else:
        mixture_derivative = np.transpose(unsummed_mixture_derivative, (1, 0, 2)).T

    image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0]),
                                        utils.evaluate_image(omega_coord, image_gradient[1])])
    prod_2 = coeff_ * mixture_derivative
    grad = gradient_gauss_energy_for_each_vertex(prod_2.T, affine_omega_coord, image_gradient_by_point)
    return grad


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    dx = np.array([image_gradient_by_point[0]])
    dy = np.array([image_gradient_by_point[1]])

    x_ = aux.T * dx
    y_ = aux.T * dy
    second_prod_x = np.dot(np.transpose(x_, (2, 0, 1)), affine_omega_coord)
    second_prod_y = np.dot(np.transpose(y_, (2, 0, 1)), affine_omega_coord)
    grad = np.concatenate([second_prod_x, second_prod_y], axis=1).T
    return grad