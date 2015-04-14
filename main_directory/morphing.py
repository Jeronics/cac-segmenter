__author__ = 'jeroni'

import ctypes_utils
import time
import sys

import numpy as np
from utils import *
import energyutils
import energies
from main_directory import utils
from scipy import interpolate

eagle_mask = utils.MaskClass()
eagle_mask.read_png('../dataset/eagle/eagle2/results/result16_1.05.png')
# eagle_mask.plot_image()

eagle_cage = utils.CageClass()
eagle_cage.read_txt('../dataset/eagle/eagle2/results/cage_16_1.05_out.txt')


pear_cage = utils.CageClass()
pear_cage.read_txt('../dataset/pear/pear1/results/cage_16_1.05_out.txt')

pear_image = utils.ImageClass()
pear_image.read_png('../dataset/pear/pear1/pear1.png')


end_mask = eagle_mask.mask

eagle_coord = np.array(np.where(end_mask == 255.)).transpose()
print eagle_coord


affine_contour_coordinates = ctypes_utils.get_affine_contour_coordinates(eagle_coord.copy(), eagle_cage.cage)
transformed_coord = np.dot(affine_contour_coordinates, pear_cage.cage)
print transformed_coord

pear_coord = transformed_coord.astype(int)
print pear_coord
values = pear_image.image[pear_coord[:,0], pear_coord[:, 1]]

print ''
print values
end_image=utils.ImageClass(np.zeros([eagle_mask.shape[0],eagle_mask.shape[1],3]))
end_image.image[np.where(end_mask == 255)]=values

end_image.plot_image()




