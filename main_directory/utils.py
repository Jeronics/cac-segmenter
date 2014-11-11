__author__ = 'jeronicarandellsaladich'

import re
import sys
import numpy as np
from scipy import *
from matplotlib import pyplot
from scipy import ndimage
import scipy
from scipy import misc
import PIL
import math

import matplotlib.pyplot as plt
from PIL import Image

########### VISUALITON
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def read_png(name):
    """Return image data from a raw PNG file as numpy array.
    """
    im = scipy.misc.imread(name)
    im = im.astype(np.float64)

    return im

def printNpArray(im):
    im = im.astype('uint8')
    plt.gray()
    plt.imshow(im, interpolation='nearest')
    plt.axis('off')
    plt.show()

def binarizePgmImage(im):
    for i in xrange(0, im.shape[0]):
        for j in xrange(0, im.shape[1]):
            if im[i, j] >= 125.:
                im[i, j] = 155.
            else:
                im[i, j] = 0.
    return im

def plotContourOnImage(contour_coordinates,image):
    matriu = contour_coordinates.astype(int)
    matriu = np.fliplr(matriu)
    image_copy = np.copy(image)

    image_r = image_copy[:, :, 0]
    image_g = image_copy[:, :, 1]
    image_b = image_copy[:, :, 2]
    size = image_r.shape
    for a in matriu:
        if (is_inside_image(a, size)):
            image_r[a[0]][a[1]] = 255.
            image_g[a[0]][a[1]] = 255.
            image_b[a[0]][a[1]] = 255.

    image_copy[:, :, 0] = image_r
    image_copy[:, :, 1] = image_g
    image_copy[:, :, 2] = image_b

    printNpArray(image_copy)


def get_inputs(arguments):
    """Return imagage, cage/s and mask from input as numpy array.
        Specification:  ./cac model imatge mascara caixa_init [caixa_curr]
    """

    if (len(arguments) != 6 and len(arguments) != 5 ):
        print 'Wrong Use!!!! Expected Input ' +sys.argv[0] + ' model(int) image(int) mask(int) init_cage(int) [curr_cage(int)]'
        sys.exit(1)

    model = arguments[0]
    mask = int(arguments[1])
    init_cage = int(arguments[2])
    if len(arguments) == 6:
        curr_cage = int(arguments[5])
    else:
        curr_cage = None

    folder_name = 'my_folder'
    # PATHS
    test_path = r'../test/'+folder_name+'/'
    mask_num = '%(number)02d' % {"number": mask}
    init_cage_name = '%(number)02d' % {"number": init_cage}
    curr_cage_name = '%(number)02d' % {"number": curr_cage}

    image_name = test_path + 'image' + '.png'
    mask_name = test_path + 'mask_' + mask_num + '.pgm'  # Both .pgm as well as png work. png gives you a rbg image!
    init_cage_name = test_path + 'cage_'+init_cage_name+'.txt'
    curr_cage_name = test_path + 'cage_'+curr_cage_name+'.txt'

    # LOAD Cage/s and Mask
    image = read_png(image_name)
    mask_file = read_png(mask_name)
    mask_file = binarizePgmImage(mask_file)
    init_cage_file = np.loadtxt(init_cage_name, float)
    curr_cage_file = np.loadtxt(curr_cage_name, float)

    return image, mask_file, init_cage_file, curr_cage_file


def evaluate_image(coordinates, image):
    '''
    Evaluates image
    :param coordinates: index numpy array
    :param image: numpy array image
    :return:
        Result of image, when indexes are not inside the image return maximum 255
    '''
    image_evaluations = np.ones([1,len(coordinates)])*255
    image_evaluations=image_evaluations[0]
    coordinates_booleans = are_inside_image(coordinates, image.shape)
    coordinates = coordinates[coordinates_booleans]
    coordinates = np.transpose(coordinates).tolist()
    image_evaluations[coordinates_booleans] = image[coordinates]
    return image_evaluations

# Check if list of points are inside an image given only the shape.
def are_inside_image(coordinates, size):
    boolean = (coordinates[:, 0] > -1) & (coordinates[:, 0] < size[0]) & (coordinates[:, 1] > -1) & (coordinates[:, 1] < size[1])
    return boolean
# TODO: coordinates[coordinates[:,0]>1] does not work

#Checks if point is inside an image given only the shape of the image.
def is_inside_image(a, size):
    if a[0] >= 0 and a[0] < size[0] and a[1] >= 0 and a[1] < size[1]:
        return True
    else:
        return False