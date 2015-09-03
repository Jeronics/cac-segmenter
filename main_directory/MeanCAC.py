from CACSegmenter import CACSegmenter
from CAC import CAC
import numpy as np
import energy_utils_mean as mean_energy
import utils
from scipy.ndimage.filters import gaussian_filter


class MeanCAC(CAC):
    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        image = image_obj.gray_image
        omega_1 = mean_energy.mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
        omega_2 = mean_energy.mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
        energy = (omega_1 + omega_2) / float(2)
        return energy

    def energy_gradient(self, omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image_obj):
        # Calculate Image gradient
        image = image_obj.gray_image
        image = gaussian_filter(image, sigma=0.5, order=0)
        image_gradient = np.array(np.gradient(image))

        # Calculate Energy:
        print 'interior'
        omega_1 = mean_energy.mean_energy_grad_per_region(omega1_coord, affine_omega_1_coord, image, image_gradient)
        print 'exterior'
        omega_2 = mean_energy.mean_energy_grad_per_region(omega2_coord, affine_omega_2_coord, image, image_gradient)
        energy = omega_1 + omega_2
        print '\n'
        return energy

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        utils.plotContourOnImage(contour_coord, image_obj.gray_image, points=cage_obj.cage, color=color,
                                 points2=cage_obj.cage - alpha * 10 * grad_k)


if __name__ == '__main__':
    mean_gray_cac = CACSegmenter(MeanCAC)
    parameter_list = mean_gray_cac.get_parameters()

    dataset = mean_gray_cac.load_dataset('synthetic_input.txt')
    results_folder = 'segment_results_synthetic/' + mean_gray_cac.CAC.__name__
    mean_gray_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=True)
    # color_cac.train_model('BSDS300_input.txt')