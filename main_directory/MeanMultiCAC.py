from CACSegmenter import CACSegmenter
import numpy as np
import energies
import utils


class MeanMultiCAC(CACSegmenter):
    def __init__(self, type, weight):
        CACSegmenter.__init__(self)
        self.energy = 1
        self.parameters['other'] = [1, 2, 3, 4]
        self.type = type
        self.weight = weight

    def mean_energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        image = image_obj.gray_image
        omega_1 = energies.mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
        omega_2 = energies.mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
        energy = (omega_1 + omega_2) / float(2)
        return energy

    def mean_energy_grad(self, omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        # Calculate Image gradient
        image = image_obj.gray_image
        image_gradient = np.array(np.gradient(image))

        # Calculate Energy:
        omega_1 = energies.mean_energy_grad_per_region(omega1_coord, affine_omega_1_coord, image, image_gradient)
        omega_2 = energies.mean_energy_grad_per_region(omega2_coord, affine_omega_2_coord, image, image_gradient)
        energy = omega_1 + omega_2
        return energy

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=color,
                                 points2=cage_obj.cage - alpha * 10 * grad_k)


if __name__ == '__main__':
    rgb_cac = MeanMultiCAC(['C', 'N', 'N'], [1, 1, 1])
    parameter_list = rgb_cac.get_parameters()

    dataset = rgb_cac._load_dataset('BSDS300_input.txt')
    results_folder = 'segment_results'
    rgb_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=False)
    # color_cac.train_model('BSDS300_input.txt')