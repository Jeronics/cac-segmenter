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

# ########## VISUALITON
def rgb2gray(rgb):
    if len(rgb.shape) == 3:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = rgb
    return gray


def read_png(name):
    """Return image data from a raw PNG file as numpy array.
    """
    im = scipy.misc.imread(name)
    im = im.astype(np.float64)
    return im


def printNpArray(im, show_plot=True):
    im_aux = im.astype('uint8')
    plt.gray()
    plt.imshow(im_aux, interpolation='nearest')
    plt.axis('off')
    if show_plot:
        plt.show()


def binarizePgmImage(image):
    im = image
    for i in xrange(0, im.shape[0]):
        for j in xrange(0, im.shape[1]):
            if im[i, j] >= 125.:
                im[i, j] = 155.
            else:
                im[i, j] = 0.
    return im


def plotContourOnImage(contour_coordinates, image, points=[], color=[255., 255., 255.]):
    matriu = contour_coordinates.astype(int)
    matriu = np.fliplr(matriu)
    image_copy = np.copy(image)

    if len(image.shape) == 3:
        image_r = image_copy[:, :, 0]
        image_g = image_copy[:, :, 1]
        image_b = image_copy[:, :, 2]
        size = image.shape
    else:
        image_gray = image_copy
        size = image_gray.shape

    for a in matriu:
        if is_inside_image(a, size):
            if len(size) == 3:
                image_r[a[0]][a[1]] = color[0]
                image_g[a[0]][a[1]] = color[1]
                image_b[a[0]][a[1]] = color[2]
            else:
                image_gray[a[0]][a[1]] = 255.

    if len(size) == 3:
        image_copy[:, :, 0] = image_r
        image_copy[:, :, 1] = image_g
        image_copy[:, :, 2] = image_b
    else:
        image_copy = image_gray
    printNpArray(image_copy, False)
    if points != []:
        points = np.concatenate((points, [points[0]]))
        points = np.transpose(points)
        plt.scatter(points[0], points[1], marker='o', c='b', )
        plt.plot(points[0], points[1])
    plt.show()


def get_inputs(arguments):
    """Return imagage, cage/s and mask from input as numpy array.
        Specification:  ./cac model imatge mascara caixa_init [caixa_curr]
    """

    if (len(arguments) != 6 and len(arguments) != 5 ):
        print 'Wrong Use!!!! Expected Input ' + sys.argv[0] + \
              ' model(int) image(int) mask(int) init_cage(int) [curr_cage(int)]'
        sys.exit(1)

    model = arguments[0]
    mask = int(arguments[1])
    init_cage = int(arguments[2])
    if len(arguments) == 6:
        curr_cage = int(arguments[5])
    else:
        curr_cage = None

    folder_name = 'ovella'
    # PATHS
    test_path = r'../test/' + folder_name + '/'
    mask_num = '%(number)02d' % {"number": mask}
    init_cage_name = '%(number)02d' % {"number": init_cage}
    curr_cage_name = '%(number)02d' % {"number": curr_cage}

    image_name = test_path + 'image' + '.png'
    mask_name = test_path + 'mask_' + mask_num + '.png'  # Both .pgm as well as png work. png gives you a rbg image!
    init_cage_name = test_path + 'cage_' + init_cage_name + '.txt'
    curr_cage_name = test_path + 'cage_' + curr_cage_name + '.txt'

    # LOAD Cage/s and Mask
    image = read_png(image_name)
    mask_file = read_png(mask_name)
    mask_file = binarizePgmImage(mask_file)
    init_cage_file = np.loadtxt(init_cage_name, float)
    curr_cage_file = np.loadtxt(curr_cage_name, float)

    # FROM RGBA to RGB if necessary
    if image.shape[2] == 4:
        image = image[:, :, 0:3]
    return image, mask_file, init_cage_file, curr_cage_file


def evaluate_image(coordinates, image, outside_value=255.):
    '''
    Evaluates image
    :param coordinates: index numpy array
    :param image: numpy array image
    :return:
        Result of image, when indexes are not inside the image return maximum 255
    '''
    image_evaluations = np.ones([1, len(coordinates)]) * outside_value
    image_evaluations = image_evaluations[0]
    coordinates_booleans = are_inside_image(coordinates, image.shape)
    coordinates_aux = coordinates[coordinates_booleans]
    coordinates_aux = np.transpose(coordinates_aux).tolist()
    image_evaluations[coordinates_booleans] = image[coordinates_aux]
    return image_evaluations


def evaluate_bilinear_interpolation(coordinates, image, outside_value=255.):
    '''

    :param coordinates:
    :param image:
    :param outside_value:
    :return:
    '''
    image_with_border = np.insert(image, (0, image.shape[1]), outside_value, axis=1)
    image_with_border = np.insert(image_with_border, (0, image_with_border.shape[0]), 255, axis=0)
    evaluated_values = [bilinear_interpolate(image_with_border, coord[0] + 1, coord[1] + 1) for coord in coordinates]
    return evaluated_values


# Check if list of points are inside an image given only the shape.
def are_inside_image(coordinates, size):
    boolean = (coordinates[:, 0] > -1) & (coordinates[:, 0] < size[0]) & (coordinates[:, 1] > -1) & (
    coordinates[:, 1] < size[1])
    return boolean


# TODO: coordinates[coordinates[:,0]>1] does not work

# Checks if point is inside an image given only the shape of the image.
def is_inside_image(a, size):
    if a[0] >= 0 and a[0] < size[0] and a[1] >= 0 and a[1] < size[1]:
        return True
    else:
        return False


def bilinear_interpolate(im, y, x):
    # TODO:PROPERLY CHANGE COLUMN AND ROWS TO ROW;COL FORMAT INSTEAD OF COL;ROW
    x_aux = np.asarray(x)
    y_aux = np.asarray(y)

    x0 = np.floor(x_aux).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_aux).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y_aux)
    wb = (x1 - x) * (y_aux - y0)
    wc = (x - x0) * (y1 - y_aux)
    wd = (x - x0) * (y_aux - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id