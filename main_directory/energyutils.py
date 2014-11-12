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
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord = omega_coord[omega_boolean]
    omega_intensity = sum(image[[omega_coord[:, 0].tolist(), omega_coord[:, 1].tolist()]])
    omega_mean = omega_intensity / (len(omega_boolean[omega_boolean == True]))
    omega_std = np.std(image[[omega_coord[:, 0].tolist(), omega_coord[:, 1].tolist()]])
    return omega_mean, omega_std


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


def gradient_energy_for_each_vertex(aux, affine_omega_coordinates, image_gradient_by_point):
    image_gradient_by_point = np.transpose(image_gradient_by_point)
    aux_1 = np.multiply(aux, np.transpose(affine_omega_coordinates))
    aux_2 = np.dot(aux_1, image_gradient_by_point)
    return aux_2

def gradientEnergy(omega1_coord, omega2_coord, affine_omega1_coordinates, affine_omega2_coordinates, image):
    # Calculate Image gradient
    image_gradient = np.array(np.gradient(image))
    # Calculate Energy:
    Omega1 = gradient_Energy_per_region(omega1_coord, affine_omega1_coordinates, image, image_gradient)
    Omega2 = gradient_Energy_per_region(omega2_coord, affine_omega2_coordinates, image, image_gradient)
    Energy = Omega1+ Omega2
    print Energy

def gradient_Energy_per_region(omega_coord, affine_omega_coordinates, image, image_gradient):
    omega_coord = omega_coord.astype(int)
    # E_mean
    mean_omega, omega_std = calculateOmegaMean(omega_coord, image)
    print mean_omega, omega_std
    aux = utils.evaluate_image(omega_coord, image) - mean_omega
    image_gradient_by_point = [utils.evaluate_image(omega_coord, image_gradient[0],0), utils.evaluate_image(omega_coord, image_gradient[1], 0)]
    return gradient_energy_for_each_vertex(aux, affine_omega_coordinates, image_gradient_by_point)
