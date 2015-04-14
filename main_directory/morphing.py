__author__ = 'jeroni'

import ctypes_utils
from utils import *
import numpy as np
from main_directory import utils
from scipy import interpolate
import matplotlib.pyplot as plt


def morphing(origin_image, origin_cage, destination_mask, destination_cage):
    end_mask = destination_mask.mask
    eagle_coord = np.array(np.where(end_mask == 255.)).transpose()
    eagle_coord_ = eagle_coord.copy()
    eagle_coord_ = eagle_coord_.astype(dtype=float64)
    affine_contour_coordinates = ctypes_utils.get_affine_contour_coordinates(eagle_coord_, destination_cage.cage)

    transformed_coord = np.dot(affine_contour_coordinates, origin_cage.cage)

    pear_coord = transformed_coord.astype(int)
    values = origin_image.image[pear_coord[:, 0], pear_coord[:, 1]]

    end_image = utils.ImageClass(np.zeros([destination_mask.shape[0], destination_mask.shape[1], 3]))
    end_image.image[np.where(end_mask == 255)] = values

    return end_image


if __name__ == '__main__':
    eagle_mask = utils.MaskClass()
    eagle_mask.read_png('../dataset/eagle/eagle2/results/result16_1.05.png')

    eagle_cage = utils.CageClass()
    eagle_cage.read_txt('../dataset/eagle/eagle2/results/cage_16_1.05_out.txt')

    pear_cage = utils.CageClass()
    pear_cage.read_txt('../dataset/pear/pear1/results/cage_16_1.05_out.txt')

    pear_image = utils.ImageClass()
    pear_image.read_png('../dataset/pear/pear1/pear1.png')

    morphed_image = morphing(pear_image, pear_cage, eagle_mask, eagle_cage)


    eagle_mask = utils.MaskClass()
    eagle_mask.read_png('../dataset/apple/apple1/results/result16_1.05.png')

    eagle_cage = utils.CageClass()
    eagle_cage.read_txt('../dataset/apple/apple1/results/cage_16_1.05_out.txt')

    morphed_image = morphing(pear_image, pear_cage, eagle_mask, eagle_cage)

    morphed_image.plot_image()