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

import matplotlib.pyplot as plt
from PIL import Image


def read_png(name):
    """Return image data from a raw PNG file as numpy array.
    """
    im = scipy.misc.imread(name)
    im = im.astype(np.float64)
    print im.shape
    return im


def printNpArray(im):
    im = im.astype('uint8')
    plt.gray()
    plt.imshow(im, interpolation='nearest')
    plt.axis('off')
    plt.show()



def get_inputs(arguments):
    """Return imagage, cage/s and mask from input as numpy array.
        Specification:  ./cac model imatge mascara caixa_init [caixa_curr]
    """

    if (len(arguments) != 6 and len(arguments) != 5 ):
        print 'Wrong Use!!!! Expected Input ' +sys.argv[0] + ' model(int) image(int) mask(int) init_cage(int) [curr_cage(int)]'
        sys.exit(1)

    for arg in arguments:
        print arg
    model = arguments[0]
    mask = int(arguments[1])
    init_cage= int(arguments[2])
    if len(arguments) == 6:
        curr_cage = int(arguments[5]);
    else:
        curr_cage = None

    folder_name='my_folder'
    # PATHS
    test_path = r'../test/'+folder_name+'/'
    mask_num = '%(number)02d' % {"number": mask}
    init_cage_name = '%(number)02d' % {"number": init_cage}
    curr_cage_name = '%(number)02d' % {"number": curr_cage}


    image_name= test_path + 'image' + '.png'
    mask_name = test_path + 'mask_' + mask_num + '.png'
    init_cage_name = test_path + 'cage_'+init_cage_name+'.txt'
    curr_cage_name = test_path + 'cage_'+curr_cage_name+'.txt'

    # LOAD Cage/s and Mask
    image = read_png(image_name)
    mask_file = read_png(mask_name)
    init_cage_file = np.loadtxt(init_cage_name, float)
    curr_cage_file = np.loadtxt(curr_cage_name, float)
    #printNpArray(image)
    return image, mask_file, init_cage_file, curr_cage_file


#TODO
