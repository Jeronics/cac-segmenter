import numpy as np

from CACSegmenter import CACSegmenter
from CAC import CAC
import energy_utils_mixture_gaussian as g_energies
import utils


class MixtureGaussianCAC(CAC):
    def __init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=None, weight=None, band_size=500):
        ground_truth_obj.plot_image()
        CAC.__init__(self, image_obj, ground_truth_obj, cage_obj, ground_truth_obj, type=type, weight=weight,
                     band_size=band_size)
        inside_gmm, outside_gmm = g_energies.mixture_initialize_seed(self)
        print 'Inside Mean', inside_gmm.means_
        print 'Inside covars', inside_gmm.covars_
        print 'Inside weights', inside_gmm.weights_

        print 'outside Mean', outside_gmm.means_
        print 'outside Covars', outside_gmm.covars_
        print 'outside weight', outside_gmm.weights_
        self.inside_gmm = inside_gmm
        self.outside_gmm = outside_gmm

    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        image = image_obj.gray_image
        omega_1 = g_energies.gauss_energy_per_region(omega_1_coord, affine_omega_1_coord, self.inside_gmm, image)
        omega_2 = g_energies.gauss_energy_per_region(omega_2_coord, affine_omega_2_coord, self.outside_gmm, image)
        energy = (omega_1 + omega_2) / float(2)
        return energy

    def energy_gradient(self, omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        # Calculate Image gradient
        image = image_obj.gray_image
        image_gradient = np.array(np.gradient(image))

        # Calculate Energy:
        print 'inside'
        omega_1 = g_energies.grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, self.inside_gmm, image,
                                                          image_gradient)
        print 'outside'
        omega_2 = g_energies.grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, self.outside_gmm, image,
                                                          image_gradient)
        print '\n'

        import energies
        # omega_1=omega_1/len(omega1_coord)
        # omega_2=omega_2/len(omega2_coord)
        # omega_1 = energies.multiple_standardize(omega_1)
        # omega_2 = energies.multiple_standardize(omega_2)

        energy = omega_1 + omega_2
        return energy

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, image_obj.gray_image, points=cage_obj.cage, color=color,
                                 points2=cage_obj.cage - alpha * 10 * grad_k)


if __name__ == '__main__':
    mixture_gaussian_gray_cac = CACSegmenter(MixtureGaussianCAC)
    parameter_list = mixture_gaussian_gray_cac.get_parameters()

    dataset = mixture_gaussian_gray_cac.load_dataset('synthetic_input.txt')
    results_folder = 'segment_results_synthetic/' + mixture_gaussian_gray_cac.CAC.__name__
    mixture_gaussian_gray_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=True)
