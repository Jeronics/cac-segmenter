from CACSegmenter import CACSegmenter
from CAC import CAC
import energy_utils_multi_mean as multi_mean_energy
import utils


class MeanMultiCAC(CAC):
    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        total_energy = []
        for slice, (t, w) in enumerate(zip(self.type, self.weight)):
            omega_1 = multi_mean_energy.generic_mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image_obj,
                                                                       t, slice)
            omega_2 = multi_mean_energy.generic_mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image_obj,
                                                                       t, slice)
            energy = (omega_1 + omega_2) / float(2)
            total_energy.append(w * energy)
        return sum(total_energy)

    def energy_gradient(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        total_grad_energy = []
        for slice, (t, w) in enumerate(zip(self.type, self.weight)):
            grad_energy_1 = multi_mean_energy.generic_grad_mean_energy_per_region(omega_1_coord, affine_omega_1_coord,
                                                                                  image_obj, t,
                                                                                  slice)
            grad_energy_2 = multi_mean_energy.generic_grad_mean_energy_per_region(omega_2_coord, affine_omega_2_coord,
                                                                                  image_obj, t,
                                                                                  slice)
            grad_energy = grad_energy_1 + grad_energy_2
            total_grad_energy.append(w * grad_energy)
        return sum(total_grad_energy)

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=color,
                                 points2=cage_obj.cage - alpha * 10 * grad_k)


if __name__ == '__main__':
    rgb_cac = CACSegmenter(MeanMultiCAC, type=['N', 'N', 'N'], weight=[0, 0, 1])
    parameter_list = rgb_cac.get_parameters()
    dataset = rgb_cac.load_dataset('BSDS300_input.txt')
    results_folder = 'segment_results/' + rgb_cac.CAC.__name__
    rgb_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=False)
