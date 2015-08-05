import numpy as np

import utils
import opencv_utils as opencv_ut
from MaskClass import MaskClass
import mixture_gaussian


'''
                        MIXTURE GAUSSIAN ENERGY
'''


def mixture_initialize_seed(CAC, from_gt=True):
    image = CAC.image_obj.gray_image
    if from_gt:
        print 'Seed from ground truth...'
        inside_mask_seed = CAC.ground_truth_obj
        outside_mask_seed = CAC.ground_truth_obj

        inside_seed = inside_mask_seed.mask
        outside_seed = 255. - outside_mask_seed.mask

        inside_seed = opencv_ut.erode(inside_seed, width=10)
        outside_seed = opencv_ut.erode(outside_seed, width=10)

    else:
        print 'Seed from mask...'
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
    # utils.printNpArray(inside_seed)
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
    values_in_region = image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]]
    values_in_region = np.array([values_in_region]).T
    # plt.figure()
    # p, bins, hist = plt.hist(values_in_region, 255)
    # plt.show()
    gmm = mixture_gaussian.get_number_of_components(values_in_region, maximum_n_components=7)
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
    energy = (omega_1 + omega_2) / 2.
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

    energy = (omega_1 + omega_2)
    return energy


def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([w for w in gmm.weights_])

    x_m = utils.evaluate_image(omega_coord, image) - means
    x_m = x_m.T
    print x_m.shape
    x_m_squared = x_m * x_m
    denom = 2 * covars
    exp_aux = np.exp(-x_m_squared / denom)
    coeff = 1 / (np.sqrt(2 * np.pi) * np.sqrt(covars))
    mixt = coeff * exp_aux
    mixture_prob = weights * mixt
    energy = -np.sum(np.log(mixture_prob))
    return energy


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([w for w in gmm.weights_])

    x_m = utils.evaluate_image(omega_coord, image) - means
    x_m = x_m.T
    print x_m.shape
    x_m_squared = x_m * x_m
    denom = 2 * covars
    exp_aux = np.exp(-x_m_squared / denom)
    coeff = 1 / (np.sqrt(2 * np.pi) * np.sqrt(covars))
    mixt = coeff * exp_aux
    mixture_prob = weights * mixt
    print mixture_prob.shape, 'hi'
    print np.sum(mixture_prob)


    # caluculate 1/P(x)
    coeff_ = 1 / mixture_prob

    # calculate the derivative of the mixture
    mixture_derivative = np.sum(mixture_prob * (- x_m / covars))

    image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0]),
                                        utils.evaluate_image(omega_coord, image_gradient[1])])

    prod_2 = coeff_ * mixture_derivative
    grad = gradient_gauss_energy_for_each_vertex(prod_2, affine_omega_coord, image_gradient_by_point)

    return grad


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    print aux[0].shape, affine_omega_coord.shape
    first_prod = aux[0] * affine_omega_coord
    print 'passed'
    second_prod = np.dot(first_prod.T, image_gradient_by_point.T)
    return second_prod