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
        total_energy = []
        for slice, (t, w) in enumerate(zip(self.type, self.weight)):
            omega_1 = self.generic_mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image_obj, t, slice)
            omega_2 = self.generic_mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image_obj, t, slice)
            energy = (omega_1 + omega_2) / float(2)
            total_energy.append(w * energy)
        return sum(total_energy)

    def mean_energy_grad(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        total_grad_energy = []
        for slice, (t, w) in enumerate(zip(self.type, self.weight)):
            grad_energy_1 = self.generic_mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image_obj, t, slice)
            grad_energy_2 = self.generic_mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image_obj, t, slice)
            grad_energy = grad_energy_1 + grad_energy_2
            total_grad_energy.append(w * grad_energy)
        return sum(total_grad_energy)

    def generic_mean_energy_per_region(self, omega_coord, affine_omega_coord, image_obj, type, slice):
        if type == 'C':
            omega_energy = energies.mean_color_energy_per_region(omega_coord, image_obj)
        if type == 'N':
            image = image_obj.image[:, :, slice]
            omega_energy = energies.mean_energy_per_region(omega_coord, affine_omega_coord, image)
        return omega_energy


    def generic_grad_mean_energy_per_region(self, omega_coord, affine_omega_coord, image_obj, type, slice):
        if type == 'C':
            omega_energy = energies.grad_mean_color_energy_per_region(omega_coord, affine_omega_coord, image_obj)
        if type == 'N':
            image = image_obj.image[:, :, slice]
            image_gradient = np.array(np.gradient(image))
            omega_energy = energies.mean_energy_grad_per_region(omega_coord, affine_omega_coord, image, image_gradient)
        return omega_energy

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