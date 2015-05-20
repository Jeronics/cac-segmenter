from CACSegmenter import CACSegmenter
import energies

class MeanColorCAC(CACSegmenter):
    def __init__(self):
        CACSegmenter.__init__(self)
        self.energy = 1
        self.parameters['other'] = [1, 2, 3, 4]

    def mean_energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
        energy1 = energies.mean_color_energy_per_region(omega_1_coord, image)
        energy2 = energies.mean_color_energy_per_region(omega_2_coord, image)
        energy = (energy1 + energy2) / 2.
        return energy


    def mean_energy_grad(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
        grad_energy_1 = energies.grad_mean_color_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
        grad_energy_2 = energies.grad_mean_color_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
        return grad_energy_1 + grad_energy_2


if __name__ == '__main__':
    color_cac = MeanColorCAC()
    parameter_list = color_cac.get_parameters()
    dataset = color_cac._load_dataset('BSDS300_input.txt')
    color_cac.test_model(dataset, parameter_list[0])
    # color_cac.train_model('BSDS300_input.txt')