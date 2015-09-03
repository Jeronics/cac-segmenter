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


def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image):
    print 'step 2'
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([1 for w in gmm.weights_])
    x_m = utils.evaluate_image(omega_coord, image) - means
    x_m = x_m.T
    x_m_squared = x_m * x_m
    denom = 2 * covars
    exp_aux = np.exp(x_m_squared / denom)
    coeff = 1 / (np.sqrt(2 * np.pi) * np.sqrt(covars))
    mixt = coeff * exp_aux
    mixture_prob = weights * mixt
    energy = np.sum(np.log(mixture_prob))
    print 'Step 2'
    return energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_]).T
    weights = np.array([w for w in gmm.weights_]).T

    x_ = utils.evaluate_image(omega_coord, image)
    x_m = np.tile(x_, (2, 1, 1)) - np.transpose(np.tile(means, (len(x_), 1, 1)), (1, 0, 2))
    # print x_m.shape
    x_m_squared = x_m * x_m
    denom = np.linalg.inv(2*covars)

    aux = -np.dot(x_m_squared, denom)
    exp_aux = np.exp(aux)


    import pdb;

    pdb.set_trace()
    coeff = 1 / (np.sqrt(2 * np.pi) * np.sqrt(covars.T))
    mixt = coeff * exp_aux
    unsummed_mixture_prob = weights * mixt

    if len(means) > 1:
        mixture_prob = np.array([np.sum(unsummed_mixture_prob, axis=1)]).T
    else:
        mixture_prob = unsummed_mixture_prob

    # caluculate 1/P(x)
    coeff_ = 1 / mixture_prob

    print 'energy', np.sum(np.log(mixture_prob))
    # calculate the derivative of the mixture
    unsumed_mixture_derivative = unsummed_mixture_prob * (x_m / covars.T)
    if len(means) > 1:
        mixture_derivative = np.array([np.sum(unsumed_mixture_derivative, axis=1)]).T
    else:
        mixture_derivative = unsumed_mixture_derivative

    image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0]),
                                        utils.evaluate_image(omega_coord, image_gradient[1])])
    prod_2 = coeff_ * mixture_derivative
    grad = gradient_gauss_energy_for_each_vertex(prod_2, affine_omega_coord, image_gradient_by_point)
    return grad


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    dx = np.array([image_gradient_by_point[0]]).T
    dy = np.array([image_gradient_by_point[1]]).T
    x_ = aux * dx
    y_ = aux * dy
    second_prod_x = np.dot(np.transpose(x_, (2, 0, 1)), affine_omega_coord)
    second_prod_y = np.dot(np.transpose(y_, (2, 0, 1)), affine_omega_coord)
    grad = np.concatenate([second_prod_x, second_prod_y]).T
    return grad



    # ================================
    #
    # def grad_gauss_energy(omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    # '''
    # Computes the derivative of the Gaussian Energy of an Image with respect to the control points
    #     :param omega1_coord (numpy array): Omega coordinates for region Omega 1
    #     :param omega2_coord (numpy array): Omega coordinates for region Omega 2
    #     :param affine_omega_1_coord (numpy array): Affine coordinates for region Omega 1
    #     :param affine_omega_2_coord (numpy array): Affine coordinates for region Omega 2
    #     :param image (numpy array): The Image
    #     :return:
    #     '''
    #     # Calculate Image gradient
    #     image_gradient = np.array(np.gradient(image))
    #
    #     # Calculate Energy Per region:
    #     omega_1 = grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, self.gmm, image, image_gradient)
    #     omega_2 = grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, self.gmm, image, image_gradient)
    #
    #     energy = -(omega_1 + omega_2)
    #     return energy
    #
    #
    # def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image):
    #     region_energy = 0
    #     for i, (omega_mean, omega_std, omega_weight) in enumerate(zip(gmm.means_, gmm.covars_, gmm.weights_)):
    #         aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    #         k = len(omega_std)
    #         term_1 = len(aux) * k * np.log(2 * np.pi)
    #         term_2 = len(aux) * np.log(np.linalg.det(omega_std))
    #         sigma = np.linalg.inv(omega_std)
    #         x_sigma = np.dot(aux, sigma)
    #         term_3 = np.multiply(x_sigma, aux)
    #         term_3 = np.sum(term_3, axis=1)
    #         region_energy += omega_weight * (term_1 + term_2 + sum(term_3))
    #     return region_energy
    #
    #
    # def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient):
    #     grad = np.zeros([affine_omega_coord.shape[1], omega_coord.shape[1], image.shape[2]])
    #     image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0], 0),
    #                                         utils.evaluate_image(omega_coord, image_gradient[1], 0)])
    #     for i, (omega_mean, omega_std, omega_weight) in enumerate(zip(gmm.means_, gmm.covars_, gmm.weights_)):
    #         sigma = np.linalg.inv(omega_std)
    #         aux = utils.evaluate_image(omega_coord, image)
    #         aux -= omega_mean
    #         sigma_aux = np.dot(sigma, aux.T)
    #         grad_ = gradient_gauss_energy_for_each_vertex(sigma_aux, affine_omega_coord,
    #                                                       image_gradient_by_point) * omega_weight
    #         grad += np.transpose(grad_, (2, 0, 1))
    #     return grad
    #
    #
    # def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point, ):
    #     # image_gradient_by_point = np.transpose(image_gradient_by_point)
    #     aux = np.tile(aux, (2, 1, 1))
    #     first_prod = np.multiply(aux, np.transpose(image_gradient_by_point, (0, 2, 1)))
    #     second_prod = np.dot(first_prod, affine_omega_coord)
    #     return second_prod