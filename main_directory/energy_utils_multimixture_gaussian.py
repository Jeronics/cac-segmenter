import numpy as np

import utils
import opencv_utils as opencv_ut
from MaskClass import MaskClass
import mixture_gaussian
import copy


'''
                        MIXTURE GAUSSIAN ENERGY
'''


def multivariate_initialize_seed(CAC, from_gt=True, maximum_n_components=7, only_band=True,
                                 plot_coordinate_distribution=False):
    image = CAC.image_obj.image
    if from_gt:
        print 'Seed from ground truth...'
        inside_mask_seed = CAC.ground_truth_obj
        outside_mask_seed = CAC.ground_truth_obj

        inside_seed = inside_mask_seed.mask
        outside_seed = 255. - outside_mask_seed.mask

        inside_seed = opencv_ut.erode(inside_seed, width=15)
        outside_band_seed = opencv_ut.erode(outside_seed, width=15)
        if only_band:
            outside_band_seed = outside_band_seed - opencv_ut.erode(outside_seed, width=150)
    else:
        center = CAC.mask_obj.center
        radius_point = CAC.mask_obj.radius_point
        # print 'CENTER:', center
        # print 'RADIUS POINT:', radius_point
        # print 'RADIUS:', np.linalg.norm(np.array(radius_point) - np.array(center))
        radius = np.linalg.norm(np.array(radius_point) - np.array(center))

        inside_seed_omega = [center[0] + radius * 0.2, center[1]]
        outside_seed_omega = [center[0] + radius * 1.8, center[1]]

        inside_mask_seed = MaskClass()
        outside_mask_seed = MaskClass()

        inside_mask_seed.from_points_and_image(center, inside_seed_omega, image)
        outside_mask_seed.from_points_and_image(center, outside_seed_omega, image)

        inside_seed = inside_mask_seed.mask
        outside_band_seed = 255. - outside_mask_seed.mask


    # inside_mask_seed.plot_image()
    # CAC.mask_obj.plot_image()
    inside_coordinates = np.argwhere(inside_seed == 255.)
    outside_coordinates = np.argwhere(outside_band_seed == 255.)
    if plot_coordinate_distribution:
        print 'TODO'

    print 'Number of components:'
    inside_gmm = get_values_in_region(inside_coordinates, image, maximum_n_components)
    print 'Interior:\t', inside_gmm.n_components
    outside_gmm = get_values_in_region(outside_coordinates, image, maximum_n_components)
    print 'Exterior:\t', outside_gmm.n_components
    return inside_gmm, outside_gmm


def get_values_in_region(omega_coord, image, maximum_n_components=7):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    values_in_region = image[omega_coord_aux[:, 0], omega_coord_aux[:, 1]]
    gmm = mixture_gaussian.get_number_of_components(values_in_region, maximum_n_components)
    return gmm


def avoid_zero_terms(k, smallest_num=np.exp(-200)):
    if np.sum(k < smallest_num) > 0:
        k[k < smallest_num] = k[k < smallest_num] * 0 + smallest_num  # careful with when k[k>0] is empty
    return k


def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, smallest_num=np.exp(-200)):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([w for w in gmm.weights_]).T

    x_ = utils.evaluate_image(omega_coord, image)
    x_m = np.transpose(np.tile(x_, (1, 1, 1)), (1, 0, 2)) - means
    # print x_m.shape
    denom = np.linalg.inv(covars)
    # x_m_denom = np.dot(x_m, denom)
    # x_m_denom = np.array([x_m_denom[:, i, i, :] for i in xrange(len(weights))])
    x_m_denom = np.array([np.dot(X, M) for M, X in zip(denom, np.transpose(x_m, (1, 0, 2)))])
    aux = -np.sum(np.multiply(np.transpose(x_m_denom, (1, 0, 2)), x_m), axis=2) / 2.
    exp_x_m_denom_x_m = np.exp(aux)

    k = means.shape[1]
    coeff = 1 / (np.power(np.sqrt(2 * np.pi), k) * np.sqrt(np.linalg.det(covars)))
    mixt = exp_x_m_denom_x_m * coeff
    mixture_prob = mixt * weights
    if len(means) > 1:
        mixture_prob = np.sum(mixture_prob, axis=1)
    else:
        mixture_prob = mixture_prob
    mixture_prob = avoid_zero_terms(mixture_prob, smallest_num=smallest_num)
    energy = np.sum(np.log(mixture_prob))
    return energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient,
                                 smallest_num=np.exp(-200)):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([w for w in gmm.weights_]).T

    x_ = utils.evaluate_image(omega_coord, image)
    x_m = np.transpose(np.tile(x_, (1, 1, 1)), (1, 0, 2)) - means

    denom = np.linalg.inv(covars)
    # x_m_denom = np.dot(x_m, denom)
    # x_m_denom = np.array([x_m_denom[:, i, i, :] for i in xrange(len(weights))])
    x_m_denom = np.array([np.dot(X, M) for M, X in zip(denom, np.transpose(x_m, (1, 0, 2)))])

    aux = -np.sum(np.multiply(np.transpose(x_m_denom, (1, 0, 2)), x_m), axis=2) / 2.
    exp_x_m_denom_x_m = np.exp(aux)

    k = means.shape[1]
    coeff = 1 / (np.power(np.sqrt(2 * np.pi), k) * np.sqrt(np.linalg.det(covars)))
    mixt = exp_x_m_denom_x_m * coeff
    unsummed_mixture_prob = mixt * weights

    mixture_prob = np.sum(unsummed_mixture_prob, axis=1)

    mixture_prob = avoid_zero_terms(mixture_prob, smallest_num=smallest_num)

    # caluculate 1/P(x)
    coeff_ = 1 / mixture_prob

    denom_x_m = np.array([np.dot(M, X) for M, X in zip(denom, np.transpose(x_m, (1, 2, 0)))])
    # x_m_denom = np.transpose(np.array([np.dot(X, M) for M, X in zip(denom, np.transpose(x_m, (1, 0, 2)))]),(0,2,1))

    image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0]),
                                        utils.evaluate_image(omega_coord, image_gradient[1])])
    image_gradient_by_point = np.transpose(image_gradient_by_point, (0, 2, 1))

    denom_x_m_derImage = -np.array([np.sum(Mx * image_gradient_by_point, axis=1) for Mx in denom_x_m])/2.

    prod_2 = coeff_ * unsummed_mixture_prob.T

    prod_3 = np.multiply(prod_2, np.transpose(denom_x_m_derImage, (1, 0, 2)))
    prod_4 = np.sum(prod_3, axis=1)
    grad = np.dot(prod_4, affine_omega_coord).T
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