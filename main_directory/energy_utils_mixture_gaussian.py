import numpy as np
import matplotlib.pyplot as plt

import utils
import opencv_utils as opencv_ut
from MaskClass import MaskClass
import mixture_gaussian
from astroML.plotting import hist
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True)


'''
                        MIXTURE GAUSSIAN ENERGY UTILS
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
    inside_gmm, in_values_in_region = get_values_in_region(inside_coordinates, image)
    print 'Interior:\t', inside_gmm.n_components
    outside_gmm, out_values_in_region = get_values_in_region(outside_coordinates, image)
    print 'Exterior:\t', outside_gmm.n_components
    plot_gmm(inside_gmm, in_values_in_region, outside_gmm, out_values_in_region, name=CAC.image_obj.spec_name)
    return inside_gmm, outside_gmm


def get_values_in_region(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    values_in_region = image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]]
    values_in_region = np.array([values_in_region]).T
    gmm = mixture_gaussian.get_number_of_components(values_in_region, maximum_n_components=7)

    return gmm, values_in_region


def plot_gmm(inside_gmm, in_values_in_region, outside_gmm, out_values_in_region, name=''):

    fig = plt.figure()#figsize=(5, 5))
    # fig.subplots_adjust(bottom=0.08, top=0.95, right=0.95, hspace=0.1)
    N_values = (500, 5000)
    subplots = (211, 212)
    values = (in_values_in_region, out_values_in_region)
    gmms = [inside_gmm, outside_gmm]
    ax = fig.add_subplot(111)
    colors=('blue','red')
    region = ('interior region','exterior region')
    prop = [len(in_values_in_region), len(out_values_in_region)]
    prop = np.array(prop) /float(np.sum(prop))

    for gmm, col, val, reg, p in zip(gmms, colors, values, region, prop):
        # ax = fig.add_subplot(subplot)
        xN = values


        # Compute density via Gaussian Mixtures
        # we'll try several numbers of clusters
        t = np.array([np.linspace(0,255, 1000)])
        logprob, responsibilities = gmm.score_samples(t.T)
        # import pdb;pdb.set_trace()

        # plot the results
        # ax.plot(xN, -0.005 * np.ones(len(xN)), '|k', lw=1.5)
        # hist(values, bins='blocks', ax=ax, normed=True, zorder=1,
        #      histtype='stepfilled', lw=1.5, color='k', alpha=0.2,
        #      label="Bayesian Blocks")
        t = np.linspace(0, 255, 1000)
        ax.plot(t, np.exp(logprob)*p, '-', color=col,
                label="Mixture Model of the %s \n(%i components)" % (reg, gmm.n_components))

        # label the plot
        # ax.text(0.02, 0.95, "%i points" % len(val), ha='left', va='top',
        #         transform=ax.transAxes)
        ax.set_ylabel('Probability over the both seeds.')
        ax.set_xlabel('Intensity value')
        ax.legend(loc='upper right')

        # if subplot == 212:
        #     ax.set_xlabel('$x$')

        ax.set_xlim(0, 255)
        ax.set_ylim(-0.01, 0.06)
    foldername='../../mask_results/experiment2/gmm/'+name+'_'
    plt.savefig(foldername+'density_function.png')
    # plt.show()


def gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v for v in gmm.covars_])
    weights = np.array([np.round(w) for w in gmm.weights_])

    x_m = utils.evaluate_image(omega_coord, image) - means
    x_m = x_m.T
    x_m_squared = x_m * x_m
    denom = 2 * covars
    exp_aux = np.exp(-x_m_squared / denom)
    coeff = 1 / (np.sqrt(2 * np.pi) * np.sqrt(covars))
    mixt = coeff * exp_aux
    mixture_prob = weights * mixt

    # Avoid logarithm of zero
    mixture_prob = avoid_zero_terms(mixture_prob)

    energy = np.sum(np.log(mixture_prob))
    return energy


def avoid_zero_terms(k):
    smallest_num = np.exp(-100)
    if len(k < smallest_num) > 0:
        k[k < smallest_num] = smallest_num  # careful with when k[k>0] is empty
    return k


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, gmm, image, image_gradient):
    means = np.array([m for m in gmm.means_])
    covars = np.array([v[0] for v in gmm.covars_])
    weights = np.array([w for w in gmm.weights_])

    x_m = utils.evaluate_image(omega_coord, image).T - means
    x_m = x_m.T
    # print x_m.shape
    x_m_squared = x_m * x_m
    denom = 2 * covars.T
    exp_aux = np.exp(-x_m_squared / denom)

    coeff = 1 / (np.sqrt(2 * np.pi) * np.sqrt(covars.T))
    mixt = coeff * exp_aux

    unsummed_mixture_prob = weights * mixt

    number_of_components = len(means)
    if number_of_components > 1:
        mixture_prob = np.array([np.sum(unsummed_mixture_prob, axis=1)]).T
    else:
        mixture_prob = unsummed_mixture_prob

    # Avoid logarithm of zero
    mixture_prob = avoid_zero_terms(mixture_prob)

    energy = np.sum(np.log(mixture_prob))

    # caluculate 1/P(x)
    coeff_ = 1 / mixture_prob

    # calculate the derivative of the mixture
    unsummed_mixture_derivative = unsummed_mixture_prob * (x_m / covars.T)
    if number_of_components > 1:
        mixture_derivative = np.array([np.sum(unsummed_mixture_derivative, axis=1)]).T
    else:
        mixture_derivative = unsummed_mixture_derivative

    image_gradient_by_point = np.array([utils.evaluate_image(omega_coord, image_gradient[0]),
                                        utils.evaluate_image(omega_coord, image_gradient[1])])
    prod_2 = coeff_ * mixture_derivative
    grad = gradient_gauss_energy_for_each_vertex(prod_2, affine_omega_coord, image_gradient_by_point)

    plot_grad = False
    if plot_grad:
        show_direction(image, prod_2, omega_coord, affine_omega_coord, image_gradient_by_point)
    return grad, energy


def show_direction(image, aux, omega_coord, affine_omega_coord, image_gradient_by_point):
    dx = np.array([image_gradient_by_point[0]]).T
    dy = np.array([image_gradient_by_point[1]]).T
    x_ = aux * dx
    y_ = aux * dy
    import copy

    im = copy.copy(image)
    x = np.multiply(y_.T, affine_omega_coord[:, 3])

    import matplotlib.pyplot as plt


    # max_num = 100000
    # x[abs(x) > max_num] = x[abs(x) > max_num] / np.abs(x[abs(x) > max_num])

    x = (x - np.min(x)) / float(np.max(x) - np.min(x))
    im = im * 0.
    im[np.array(omega_coord[:, 0], dtype=int), np.array(omega_coord[:, 1], dtype=int)] = x
    plot_grad = False
    if plot_grad:
        plt.figure()
        plt.plot(im[150, :].T * 255., label="direction")
        plt.plot(image[150, :].T, label="Image")
        plt.xlabel('pixels')
        plt.ylabel('Intensity')
        plt.legend()
        plt.ylim([-1, 256])
        plt.show()
    print 'Wait a un minute .. printing the gradient.. '

    plt.figure()
    plt.imshow(im)
    plt.show()


def gradient_gauss_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point):
    dx = np.array([image_gradient_by_point[0]]).T
    dy = np.array([image_gradient_by_point[1]]).T
    x_ = aux * dx
    y_ = aux * dy
    second_prod_x = np.dot(x_.T, affine_omega_coord)
    second_prod_y = np.dot(y_.T, affine_omega_coord)
    grad = np.concatenate([second_prod_x, second_prod_y]).T
    return grad