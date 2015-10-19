from CACSegmenter import CACSegmenter
from CAC import CAC
import energy_utils_mean_hue_seed as energy_ut
import numpy as np
import utils


class HueMeanSeedCAC(CAC):
    def __init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=None, weight=None, band_size=500):
        CAC.__init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=type, weight=weight,
                     band_size=band_size)
        inside_seed_mean, outside_seed_mean = energy_ut.initialize_seed(self)
        self.inside_seed_mean = inside_seed_mean
        self.outside_seed_mean = outside_seed_mean

    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        energy1 = energy_ut.mean_color_energy_per_region(omega_1_coord, self.image_obj, self.inside_seed_mean)
        energy2 = energy_ut.mean_color_energy_per_region(omega_2_coord, self.image_obj, self.outside_seed_mean)
        energy = (energy1 + energy2) / 2.
        return energy

    def energy_gradient(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        grad_energy_1 = energy_ut.grad_mean_color_energy_per_region(omega_1_coord, affine_omega_1_coord, self.image_obj,
                                                                    self.inside_seed_mean)
        grad_energy_2 = energy_ut.grad_mean_color_energy_per_region(omega_2_coord, affine_omega_2_coord, self.image_obj,
                                                                    self.outside_seed_mean)
        return grad_energy_1 + grad_energy_2

    def _plotContourOnImage(self, contour_coord, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        # hsi_image[:, :, 0] / (2 * 3.14) * 255.
        utils.plotContourOnImage(contour_coord, self.image_obj.image,
                                 points=cage_obj.cage,
                                 color=color,
                                 points2=cage_obj.cage - alpha * 10 * grad_k)
        # self.image_obj.plot_hsi_image()


if __name__ == '__main__':
    input_filename = 'BSDS300_input.txt'
    output_folder = 'seg_im/'
    color_cac = CACSegmenter(HueMeanSeedCAC)
    parameter_list = color_cac.get_parameters()
    new_parameters = {
        'num_points': [12],
        'ratio': [1.05],
        'smallest_number': [np.exp(-200)],
    }
    color_cac.parameters = new_parameters
    color_cac.sigma = 0.25
    parameter_list = color_cac.get_parameters()



    dataset = color_cac.load_dataset(input_filename)
    results_folder = output_folder + color_cac.CAC.__name__+ '/'
    color_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=True)
