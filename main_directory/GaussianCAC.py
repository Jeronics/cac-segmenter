import numpy as np

from CACSegmenter import CACSegmenter
from CAC import CAC
import energy_utils_gaussian as g_energies
import utils
from scipy.ndimage.filters import gaussian_filter


class GaussianCAC(CAC):
    def __init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=None, weight=None, band_size=500):
        CAC.__init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=type, weight=weight,
                     band_size=band_size)
        inside_seed_mean, inside_seed_std, outside_seed_mean, outside_seed_std = g_energies.initialize_seed(self)
        self.inside_seed_mean = inside_seed_mean
        self.inside_seed_std = inside_seed_std
        self.outside_seed_mean = outside_seed_mean
        self.outside_seed_std = outside_seed_std

    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        image = self.image_obj.gray_image
        omega_1 = g_energies.gauss_energy_per_region(omega_1_coord, affine_omega_1_coord, self.inside_seed_mean,
                                                     self.inside_seed_std, image)
        omega_2 = g_energies.gauss_energy_per_region(omega_2_coord, affine_omega_2_coord, self.outside_seed_mean,
                                                     self.outside_seed_std, image)
        energy = (omega_1 + omega_2) / 2.
        return energy

    def energy_gradient(self, omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord):
        # Calculate Image gradient
        image = self.image_obj.gray_image
        image_gradient = np.array(np.gradient(image))

        # Calculate Energy:
        omega_1 = g_energies.grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, self.inside_seed_mean,
                                                          self.inside_seed_std, image, image_gradient)
        omega_2 = g_energies.grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, self.outside_seed_mean,
                                                          self.outside_seed_std, image, image_gradient)
        energy = omega_1 + omega_2
        return energy

    def _plotContourOnImage(self, contour_coord, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, self.image_obj.gray_image, points=cage_obj.cage, color=color,
                                 points2=cage_obj.cage - alpha * 10 * grad_k)


if __name__ == '__main__':
    input_filename = 'synthetic_locality_input.txt'
    output_folder = 'experiment1/'
    gaussian_gray_cac = CACSegmenter(GaussianCAC)
    parameter_list = gaussian_gray_cac.get_parameters()
    new_parameters = {
        'num_points': [12],
        'ratio': [1.05],
    }
    gaussian_gray_cac.parameters = new_parameters
    gaussian_gray_cac.sigma = 0.5

    parameter_list = gaussian_gray_cac.get_parameters()
    dataset = gaussian_gray_cac.load_dataset(input_filename)
    results_folder = output_folder + gaussian_gray_cac.CAC.__name__+'/'
    gaussian_gray_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=True)
