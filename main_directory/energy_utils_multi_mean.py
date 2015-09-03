import numpy as np
import energy_utils_mean as mean_energies
import energy_utils_mean_hue as hue_energies
from scipy.ndimage.filters import gaussian_filter


'''
                        Multi Dimensional MEAN ENERGY
'''


def generic_mean_energy_per_region(omega_coord, affine_omega_coord, image_obj, type, slice):
    if type == 'C':
        omega_energy = hue_energies.mean_color_energy_per_region(omega_coord, image_obj)
    if type == 'N':
        image = image_obj.image[:, :, slice]
        omega_energy = mean_energies.mean_energy_per_region(omega_coord, affine_omega_coord, image)
    return omega_energy


def mean_energy_multi(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image, type, weight):
    total_energy = []
    for slice, (t, w) in enumerate(zip(type, weight)):
        omega_1 = generic_mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image, t, slice)
        omega_2 = generic_mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image, t, slice)
        energy = (omega_1 + omega_2) / float(2)
        total_energy.append(w * energy)
    return sum(total_energy)


def generic_grad_mean_energy_per_region(omega_coord, affine_omega_coord, image_obj, type, slice):
    if type == 'C':
        omega_energy = hue_energies.grad_mean_color_energy_per_region(omega_coord, affine_omega_coord, image_obj)
    if type == 'N':
        image = image_obj.image[:, :, slice]
        image_gradient = np.array(np.gradient(image))
        image = gaussian_filter(image, sigma=0.5, order=0)
        omega_energy = mean_energies.mean_energy_grad_per_region(omega_coord, affine_omega_coord, image, image_gradient)
    return omega_energy


def mean_energy_grad_multi(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image, type,
                           weight):
    total_grad_energy = []
    for slice, (t, w) in enumerate(zip(type, weight)):
        grad_energy_1 = hue_energies.grad_mean_color_energy_per_region(omega_1_coord, affine_omega_1_coord, image, t,
                                                                       slice)
        grad_energy_2 = hue_energies.grad_mean_color_energy_per_region(omega_2_coord, affine_omega_2_coord, image, t,
                                                                       slice)
        grad_energy = grad_energy_1 + grad_energy_2
        total_grad_energy.append(w * grad_energy)
    return sum(total_grad_energy)

