from CACSegmenter import CACSegmenter
import numpy as np
import energies

class MeanColorCAC(CACSegmenter):
    def __init__(self):
        CACSegmenter.__init__(self)
        self.energy = 1
        self.parameters['other'] = [1, 2, 3, 4]

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


if __name__ == '__main__':
    color_cac = MeanColorCAC()
    parameter_list = color_cac.get_parameters()

    dataset = color_cac._load_dataset('BSDS300_input.txt')
    color_cac.test_model(dataset, parameter_list[0])
    # color_cac.train_model('BSDS300_input.txt')