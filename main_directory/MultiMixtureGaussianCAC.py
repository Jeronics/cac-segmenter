import numpy as np

from CACSegmenter import CACSegmenter
from CAC import CAC
import energy_utils_multimixture_gaussian as g_energies
import utils
import time

class MultiMixtureGaussianCAC(CAC):
    def __init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=None, weight=None, band_size=500):
        CAC.__init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=type, weight=weight,
                     band_size=band_size)
        start_time = time.time()
        inside_gmm, outside_gmm = g_energies.multivariate_initialize_seed(self, maximum_n_components=7)
        print("Calculating Seed: --- %s seconds ---" % (time.time() - start_time))
        self.smallest_number = np.exp(-200)
        self.inside_gmm = inside_gmm
        self.outside_gmm = outside_gmm
        self.weight = [1, 1, 1]

    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        image = self.image_obj.image
        omega_1 = g_energies.gauss_energy_per_region(omega_1_coord, affine_omega_1_coord, self.inside_gmm, image,
                                                     smallest_num=self.smallest_number)
        omega_2 = g_energies.gauss_energy_per_region(omega_2_coord, affine_omega_2_coord, self.outside_gmm, image,
                                                     smallest_num=self.smallest_number)
        energy = -(omega_1 + omega_2) / 2.
        return energy

    def energy_gradient(self, omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord):
        # Calculate Image gradient
        image = self.image_obj.image
        image_gradient = np.array([np.array(np.gradient(image[:, :, slice])) for slice in xrange(image.shape[2])])
        image_gradient = np.transpose(image_gradient, (1, 2, 3, 0))
        # Calculate Energy:
        omega_1 = g_energies.grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, self.inside_gmm, image,
                                                          image_gradient, smallest_num=self.smallest_number)
        omega_2 = g_energies.grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, self.outside_gmm, image,
                                                          image_gradient, smallest_num=self.smallest_number)
        energy = -(omega_1 + omega_2)
        return energy

    def _plotContourOnImage(self, contour_coord, current_cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, self.image_obj.image, points=current_cage_obj.cage, color=color,
                                 points2=current_cage_obj.cage - alpha * 10 * grad_k)


if __name__ == '__main__':
    input_filename = 'AlpertGBB07_input_subtest.txt'
    output_folder = 'seg_im/'
    multi_mixture_gaussian_gray_cac = CACSegmenter(MultiMixtureGaussianCAC)
    new_parameters = {
        'num_points': [20],
        'ratio': [1.1],
        'smallest_number': [np.exp(-200)],
    }
    multi_mixture_gaussian_gray_cac.parameters = new_parameters
    multi_mixture_gaussian_gray_cac.sigma = 0.25
    parameter_list = multi_mixture_gaussian_gray_cac.get_parameters()

    dataset = multi_mixture_gaussian_gray_cac.load_dataset(input_filename)
    results_folder = output_folder + multi_mixture_gaussian_gray_cac.CAC.__name__ + '/'
    multi_mixture_gaussian_gray_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=True)
