__author__ = 'jeronicarandellsaladich'
from utils import *
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



def calculateOmegaMean(omega_coord, omega_size, image):
    omega_intensity = 0.
    for a in omega_coord:
        if( is_inside_image( a, image.shape ) ):
            omega_intensity += image[a[0]][a[1]]
        else:
            omega_size -= 1
    omega_mean = omega_intensity / omega_size

    return omega_mean

def calculateMeanEnergy( omega1_coord, omega2_coord, omega1_size, omega2_size,  image ):
    omega1Mean = calculateOmegaMean( omega1_coord, omega1_size, image )
    omega2Mean = calculateOmegaMean( omega2_coord, omega2_size, image )
    Energy1 = calcuateOmegaMeanEnergy( image, omega1Mean, omega1_coord )
    Energy2 = calcuateOmegaMeanEnergy( image, omega2Mean, omega2_coord )
    return ( Energy1 + Energy2 ) / 2


def calcuateOmegaMeanEnergy( image, omegaMean, omega_coord ):
    val = 0.
    for a in omega_coord:
        if ( is_inside_image(a, image.shape ) ):
            val += pow( ( image[a[0]][a[1]] - omegaMean ), 2 )
        # ELSE: DE MOMENT NO FER RES.
    return val

def calcuateOmegaMeanEnergyGradient( image_gradient, omegaMean, omega_coord ):
    val = 0.
    cardinal=omega_coord
    for a in omega_coord:
        if ( is_inside_image( a, image_gradient.shape ) ):
            val += pow( ( image_gradient[a[0]][a[1]] - omegaMean ), 2 )
        else:
            cardinal-=1
    return val/cardinal
