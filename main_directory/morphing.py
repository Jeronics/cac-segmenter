__author__ = 'jeroni'

from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata

import ctypes_utils
from utils import *
from ImageClass import ImageClass
from MaskClass import MaskClass
import utils

def floats_are_inside_image(coordinates, size):
    boolean = (coordinates[:, 0] > 0.) & (coordinates[:, 0] < size[0]-1.) & (coordinates[:, 1] > 0) & (
        coordinates[:, 1] < size[1]-1.)
    return boolean

def morphing(origin_image, origin_cage, destination_mask, destination_cage, whole_image=True):
    if whole_image:
        end_mask = np.ones(destination_mask.mask.shape) * 255.
    eagle_coord = np.array(np.where(end_mask == 255.)).transpose()
    eagle_coord_ = eagle_coord.copy()
    eagle_coord_ = eagle_coord_.astype(dtype=float64)

    affine_contour_coordinates = ctypes_utils.get_affine_contour_coordinates(eagle_coord_, destination_cage.cage)

    transformed_coord = np.dot(affine_contour_coordinates, origin_cage.cage)
    boolean = floats_are_inside_image(transformed_coord, origin_image.image.shape)

    # pear_coord_ = transformed_coord[boolean]
    # pear_coord = transformed_coord.round().astype(int)
    # boolean = utils.are_inside_image(pear_coord, origin_image.image.shape)
    # pear_coord_ = pear_coord[boolean]
    # values = origin_image.image[pear_coord_[:, 0], pear_coord_[:, 1]]

    end_image = ImageClass(np.ones([destination_mask.mask.shape[0], destination_mask.mask.shape[1], 3]) * 255.)

    end_image_coord = np.array(np.where(np.zeros([origin_image.image.shape[0], origin_image.image.shape[1]]) == 0.)).transpose()
    values = griddata(end_image_coord, origin_image.image[end_image_coord[:, 0], end_image_coord[:, 1]], transformed_coord[boolean], method='linear')
    aux = end_image.image[np.where(end_mask == 255)]
    aux[boolean] = values

    # import pdb;pdb.set_trace()
    end_image.image[np.where(end_mask == 255)] = aux
    return end_image


def create_intermediate_cage(origin_image, origin_cage, destination_image, destination_cage, weight, parameters):
    intermediate_size = (
        np.array(origin_image.shape) * weight + np.array(destination_image.shape) * (1 - weight)).astype(int)
    aux_cage = origin_cage.cage * weight + destination_cage.cage * (1 - weight)
    intermediate_cage = CageClass()
    intermediate_cage.cage = aux_cage
    if parameters:
        # From origin mask
        aux_im = np.ones([300, 300])
        c = [150, 150]
        p = [250, 150]
        ratio = parameters['ratio'][0]
        num_points = parameters['num_points'][0]
        initial_mask = MaskClass()
        initial_mask.from_points_and_image(c, p, aux_im)
        # initial_mask.plot_image()
        initial_cage = CageClass()
        initial_cage.create_from_points(c, p, ratio, num_points)
        # plt.figure()
        # plt.scatter(initial_cage.cage[:, 0], initial_cage.cage[:, 1])
        # plt.show()
    else:
        print 'falten parametres num points i/o ratio'
    contour_coord, contour_size = ctypes_utils.get_contour(initial_mask)
    affine_contour_coordinates = ctypes_utils.get_affine_contour_coordinates(contour_coord, initial_cage.cage)
    intermediate_contour = np.dot(affine_contour_coordinates, intermediate_cage.cage)
    omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = ctypes_utils.get_omega_1_and_2_coord(600,
                                                                                                    intermediate_contour,
                                                                                                    contour_size,
                                                                                                    intermediate_size[
                                                                                                        0],
                                                                                                    intermediate_size[
                                                                                                        1])
    aux_mask = np.zeros(intermediate_size)
    omega_1_coord = omega_1_coord.astype(int)
    aux_mask[omega_1_coord[:, 0], omega_1_coord[:, 1]] = 255.
    intermediate_mask = MaskClass()
    intermediate_mask.mask = aux_mask
    return intermediate_mask, intermediate_cage


def morphing_intermediate(origin_image, origin_cage, destination_image, destination_cage, weight, parameters=None,
                          color_power=1.):
    intermediate_mask, intermediate_cage = create_intermediate_cage(origin_image, origin_cage, destination_image,
                                                                    destination_cage, weight,
                                                                    parameters)
    # intermediate_mask.plot_image()
    origin_part_image = morphing(origin_image, origin_cage, intermediate_mask, intermediate_cage)
    destination_part_image = morphing(destination_image, destination_cage, intermediate_mask, intermediate_cage)
    weights = [weight, 1. - weight]
    power_ = color_power
    color_weights = np.power(weights, power_) / np.sum(np.power(weights, power_))
    intermediate_image_ = origin_part_image.image * color_weights[0] + destination_part_image.image * color_weights[1]
    intermediate_image = ImageClass()
    intermediate_image.image = intermediate_image_
    return intermediate_image


def morphing_mask(origin_image, origin_cage, destination_mask, destination_cage):
    end_mask = destination_mask.mask
    eagle_coord = np.array(np.where(end_mask == 255.)).transpose()
    eagle_coord_ = eagle_coord.copy()
    eagle_coord_ = eagle_coord_.astype(dtype=np.float64)
    affine_contour_coordinates = ctypes_utils.get_affine_contour_coordinates(eagle_coord_, destination_cage.cage)

    transformed_coord = np.dot(affine_contour_coordinates, origin_cage.cage)

    pear_coord = transformed_coord.astype(int)
    boolean = utils.are_inside_image(pear_coord, origin_image.mask.shape)

    pear_coord_ = pear_coord[boolean]

    values = origin_image.mask[pear_coord_[:, 0], pear_coord_[:, 1]]

    end_image = MaskClass(np.zeros([destination_mask.mask.shape[0], destination_mask.mask.shape[1]]))
    aux = end_image.mask[np.where(end_mask == 255)]
    aux[boolean] = values
    end_image.mask[np.where(end_mask == 255)] = aux

    return end_image


def smoothen_image(image_obj, sigma):
    image = gaussian_filter(image_obj.image, sigma=sigma, order=0)
    image_obj.image = image

    gray_image = gaussian_filter(image_obj.gray_image, sigma=sigma, order=0)
    image_obj.gray_image = gray_image

    return image_obj


if __name__ == '__main__':
    # Fill car 2 with car 1
    # Destination
    destination_mask = MaskClass()
    destination_mask.read_png('../../morphed_mask/car_1.png')

    destination_image = ImageClass()
    destination_image.read_png('../../morphing/car_lat_1.png')

    destination_cage = CageClass()
    destination_cage.read_txt('../../morphed_mask/car_lat_1.txt')

    # Origin
    origin_mask = MaskClass()
    origin_mask.read_png('../../morphed_mask/car_2.png')

    origin_image = ImageClass()
    origin_image.read_png('../../morphing/car_lat_2.png')

    origin_cage = CageClass()
    origin_cage.read_txt('../../morphed_mask/car_lat_2.txt')

    # morphed_image = morphing(car_1_image, car_1_cage, car_2_mask, car_2_cage)
    # morphed_image.plot_image()

    # morphed_image = morphing(car_2_image, car_2_cage, car_1_mask, car_1_cage)
    # morphed_image.plot_image()

    new_parameters = {
        'num_points': [len(destination_cage.cage)],
        'ratio': [1.05],
    }
    folder_name = '../../morphing_results/car_1_interpol_2/'
    weight = 0.01
    color_power = 1.5
    while weight < 1.:
        morphed_image = morphing_intermediate(origin_image, origin_cage, destination_image, destination_cage, weight,
                                              parameters=new_parameters, color_power=color_power)
        # morphed_image.plot_image()
        # morphed_image=smoothen_image(morphed_image, min(min(weight, 1-weight),0.2))
        morphed_image.save_image(folder_name + 'results' + ''.join(str(weight).split('.')) + '.png')
        weight += 0.02
        #
        # eagle_mask = utils.MaskClass()
        # eagle_mask.read_png('../dataset/eagle/eagle2/results/result16_1.05.png')
        #
        # eagle_cage = utils.CageClass()
        # eagle_cage.read_txt('../dataset/eagle/eagle2/results/cage_16_1.05_out.txt')
        #
        # pear_cage = utils.CageClass()
        #
        # pear_cage.read_txt('../dataset/pear/pear1/results/cage_16_1.05_out.txt')
        #
        # pear_image = utils.ImageClass()
        # pear_image.read_png('../dataset/pear/pear1/pear1.png')
        #
        # morphed_image = morphing(pear_image, pear_cage, eagle_mask, eagle_cage)
        #
        # eagle_mask = utils.MaskClass()
        # eagle_mask.read_png('../dataset/apple/apple1/results/result16_1.05.png')
        #
        # eagle_cage = utils.CageClass()
        # eagle_cage.read_txt('../dataset/apple/apple1/results/cage_16_1.05_out.txt')
        #
        # morphed_image = morphing(pear_image, pear_cage, eagle_mask, eagle_cage)
        #
        # morphed_image.plot_image()