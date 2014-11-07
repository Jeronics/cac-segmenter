__author__ = 'jeronicarandellsaladich'
import sys
sys.path.append('/cac-segmenter')
import utils
import re
import numpy as np
from scipy import *
from matplotlib import pyplot
from scipy import ndimage
import scipy
from scipy import misc
import PIL
import math


def calculateOmegaMean(omega_coord, image):
    omega_size = len(omega_coord)
    omega_coord, numIndexesNotInside = utils.are_inside_image(omega_coord, image.shape)
    omega_intensity = sum(image[[omega_coord[:, 0].tolist(), omega_coord[:, 1].tolist()]])
    omega_mean = omega_intensity / (omega_size-numIndexesNotInside)
    return omega_mean


def calculateMeanEnergy(omega1_coord, omega2_coord,  image):
    omega1Mean = calculateOmegaMean(omega1_coord, image)
    omega2Mean = calculateOmegaMean(omega2_coord, image)
    Energy1 = calcuateOmegaMeanEnergy(image, omega1Mean, omega1_coord)
    Energy2 = calcuateOmegaMeanEnergy(image, omega2Mean, omega2_coord)
    return (Energy1 + Energy2) / 2


def calcuateOmegaMeanEnergy(image, omegaMean, omega_coord):
    val = 0.
    for a in omega_coord:
        if (utils.is_inside_image(a, image.shape)):
            val += pow((image[a[0]][a[1]] - omegaMean), 2)
        # ELSE: DE MOMENT NO FER RES.
    return val


def calcuateOmegaMeanEnergyGradient(image_gradient, omegaMean, omega_coord):
    val = 0.
    cardinal = omega_coord
    for a in omega_coord:
        if (is_inside_image(a, image_gradient.shape)):
            val += pow((image_gradient[a[0]][a[1]] - omegaMean), 2)
        else:
            cardinal -= 1
    return val/cardinal