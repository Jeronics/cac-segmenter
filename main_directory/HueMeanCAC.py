from CACSegmenter import CACSegmenter
from CAC import CAC
import energy_utils_mean_hue as energy_ut
from scipy.ndimage.filters import gaussian_filter

import utils


class HueMeanCAC(CAC):
    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        energy1 = energy_ut.mean_color_energy_per_region(omega_1_coord, self.image_obj)
        energy2 = energy_ut.mean_color_energy_per_region(omega_2_coord, self.image_obj)
        energy = (energy1 + energy2) / 2.
        return energy

    def energy_gradient(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        grad_energy_1 = energy_ut.grad_mean_color_energy_per_region(omega_1_coord, affine_omega_1_coord, self.image_obj)
        grad_energy_2 = energy_ut.grad_mean_color_energy_per_region(omega_2_coord, affine_omega_2_coord, self.image_obj)
        return grad_energy_1 + grad_energy_2

    def _plotContourOnImage(self, contour_coord, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, self.image_obj.hsi_image[:, :, 0] / (2 * 3.14) * 255., points=cage_obj.cage,
                                 color=color,
                                 points2=cage_obj.cage - alpha * 10 * grad_k)
        # self.image_obj.plot_hsi_image()


if __name__ == '__main__':
    color_cac = CACSegmenter(HueMeanCAC)
    parameter_list = color_cac.get_parameters()
    parameter_list = color_cac.get_parameters()
    dataset = color_cac.load_dataset('AlpertGBB07_input.txt')
    results_folder = 'segment_results_alpert_3/' + color_cac.CAC.__name__
    color_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=False)
