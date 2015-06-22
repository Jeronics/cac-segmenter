from CACSegmenter import CACSegmenter
import numpy as np
import energy_utils_gaussian as g_energies
import utils


class GaussianCACSegmenter(CACSegmenter):
    def __init__(self):
        CACSegmenter.__init__(self)
        self.energy = 1
        self.parameters['other'] = [1, 2, 3, 4]


    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        image = image_obj.gray_image
        omega_1 = g_energies.gauss_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
        omega_2 = g_energies.gauss_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
        energy = (omega_1 + omega_2) / float(2)
        return energy

    def energy_gradient(self, omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        # Calculate Image gradient
        image = image_obj.gray_image
        image_gradient = np.array(np.gradient(image))

        # Calculate Energy:
        omega_1 = g_energies.grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, image, image_gradient)
        omega_2 = g_energies.grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, image, image_gradient)
        energy = omega_1 + omega_2
        return energy

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, image_obj.gray_image, points=cage_obj.cage, color=color,
                           points2=cage_obj.cage - alpha * 10 * grad_k)


if __name__ == '__main__':
    gaussian_gray_cac = GaussianCACSegmenter()
    parameter_list = gaussian_gray_cac.get_parameters()

    dataset = gaussian_gray_cac._load_dataset('BSDS300_input.txt')
    results_folder = 'segment_results'
    gaussian_gray_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=True)
    # color_cac.train_model('BSDS300_input.txt')