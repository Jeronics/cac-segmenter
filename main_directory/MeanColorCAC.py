from CACSegmenter import CACSegmenter
import energies
import utils

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
        print 'Color 1',
        grad_energy_1 = energies.grad_mean_color_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
        print 'Color 2',
        grad_energy_2 = energies.grad_mean_color_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
        # print grad_energy_1
        # print grad_energy_2
        return grad_energy_1 + grad_energy_2

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, image_obj.hsi_image[:, :, 0]/(2*3.14)*255., points=cage_obj.cage, color=color,
                           points2=cage_obj.cage - alpha * 10 * grad_k)
        image_obj.plot_hsi_image()



if __name__ == '__main__':
    color_cac = MeanColorCAC()
    parameter_list = color_cac.get_parameters()
    dataset = color_cac._load_dataset('BSDS300_input.txt')
    color_cac.test_model(dataset, parameter_list[0], plot_evolution=False)
    # color_cac.train_model('BSDS300_input.txt')