import numpy as np
import utils

'''
                    MEAN COLOR ENERGY
'''


def mean_color_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    energy1 = mean_color_energy_per_region(omega_1_coord, image)
    energy2 = mean_color_energy_per_region(omega_2_coord, image)
    energy = (energy1 + energy2) / 2.
    return energy


def mean_color_energy_per_region(omega_1_coord, image):
    mean = mean_color_in_region(omega_1_coord, image)
    hue_comp = image.hsi_image[omega_1_coord[:, 0].tolist(), omega_1_coord[:, 1].tolist()][:, 0]
    distance = hue_color_distance(hue_comp, mean)
    energy = sum(np.power(distance, 2))
    return energy


def mean_color_energy_grad(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    grad_energy_1 = grad_mean_color_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
    grad_energy_2 = grad_mean_color_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
    return grad_energy_1 + grad_energy_2


def grad_mean_color_energy_per_region(omega_coord, affine_omega_coord, image):
    mean = mean_color_in_region(omega_coord, image)
    hue_values = image.hsi_image[omega_coord[:, 0].tolist(), omega_coord[:, 1].tolist()][:, 0]
    directed_distances = directed_hue_color_distance(mean, hue_values)
    hue_gradient = get_hsi_derivatives(omega_coord, image)
    grad_energy = np.dot(np.multiply(affine_omega_coord.T, directed_distances), hue_gradient.T)
    return grad_energy


def hue_color_distance(hue1, hue2):
    '''
    Note that all input angles must be positive.
    :param hue1:
    :param hue2:
    :return:
    '''
    dist = hue1.copy() if np.array(hue1).size >= np.array(hue2).size else hue2.copy()
    cond1 = (np.abs(hue2 - hue1) <= np.pi)
    cond2 = False == cond1

    dist = hue1.copy()
    dist[cond1] = np.abs(hue2 - hue1)[cond1]
    dist[cond2] = 2 * np.pi - np.abs(hue2 - hue1)[cond2]
    return dist


def directed_hue_color_distance(hue1, hue2):
    '''

    :param hue1 (numpy.ndarray): must be an ndarray
    :param hue2 ():
    :return:
    '''
    dist = hue1.copy() if np.array(hue1).size >= np.array(hue2).size else hue2.copy()
    cond1 = (np.abs(hue2 - hue1) <= np.pi)
    cond2 = cond1 == False
    cond3 = (hue2 >= hue1)
    cond4 = cond3 == False

    # if isinstance(dist, float):
    # if cond1:
    # dist = hue2-hue1
    # if cond2:
    # dist = (hue2 - hue1)-2*np.pi
    #     if cond3:
    #         dist =  2 * np.pi + (hue2 - hue1)
    # else:
    dist[cond1] = (hue2 - hue1)[cond1]
    dist[cond2 & cond3] = (hue2 - hue1)[cond2 & cond3] - 2 * np.pi
    dist[cond2 & cond4] = 2 * np.pi + (hue2 - hue1)[cond2 & cond4]
    return dist


def mean_color_in_region(omega_coord, image):
    '''
    Returns the mean hue color of an image from [0, 2*Pi).
    Exception: White and Black have 0 saturation. This means that they should be avoided when doing the mean.
    :param omega_coord:
    :param affine_omega_coord:
    :param image:
    :return:
    '''

    hsi = image.hsi_image[omega_coord[:, 0].tolist(), omega_coord[:, 1].tolist()]
    hue = hsi[:, 0]
    saturation = hsi[:, 1]
    if len(hue[saturation > 0]) == 0:
        # Avoid dividing by zero
        mean_angle = 0.0
    else:
        mean_angle = np.arctan2(sum(np.sin(hue[saturation > 0])) / float(len(hue[saturation > 0])),
                                sum(np.cos(hue[saturation > 0])) / float(len(hue[saturation > 0])))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return mean_angle


def rgb_to_hsi(coordinates, image):
    '''
    Convert RGB values to HSI given the coordinates of the pixels in a region omega and the region.
    If saturation is zero, the Hue value will be 0.
    :param coordinates (numpy array): a numpy array containing coordinates in the image.
    :param omega (numpy): An image with RGB values
    :return : hue, saturation, intensity which are respectiveley:
        hue: values range from [0, 2*Pi)
        saturation: values range from [0, 255.]
        intensity: values range from [0, 255.]
    '''
    # TODO: Check if points are inside the image before accessing them. This could be done outside!
    color_list = np.transpose(image[coordinates[:, 0], coordinates[:, 1]])
    hsi_transformation = np.array([
        [1 / 3., 1 / 3., 1 / 3.],
        [1, -1 / 2., -1 / 2.],
        [0, -np.sqrt(3) / 2., np.sqrt(3) / 2.]
    ])
    hsi_ = np.dot(hsi_transformation, color_list)
    C1 = hsi_[1]
    C2 = hsi_[2]
    intensity = hsi_[0]

    # Saturation = sqrt( C1^2 + C2^2 )
    saturation = np.sqrt(sum(np.power(hsi_[1:], 2)))

    # Hue Value
    # . | ArcCos(C1/saturation)  if C2<=0
    # H=|
    # . | 2*Pi - ArcCos(C1/saturation) if C2>0
    hue = C1.copy() * 1

    # get boolean vector of C1>0
    positives = (C2 <= 0)
    negatives = (positives == False)

    # Avoid dividing by zero:
    # if saturation is 0, so is C1, then we can avoid by dividing by one.
    denom = saturation.copy()
    denom[saturation == 0.] = 1

    # Note that in one article this is incorrectly written with C2 instead of C1.
    hue[positives] = np.arccos(C1[positives] / denom[positives])
    hue[negatives] = 2 * np.pi - np.arccos(C1[negatives] / denom[negatives])

    # If saturation is zero or maximum, then hue value will be 0. This is just a notation
    hue[saturation == 0.] = 0
    return np.concatenate((np.concatenate(([hue], [saturation]), axis=0), [intensity])).T


def get_hsi_derivatives(coordinates, image):
    '''
    Gets neighboring pixels of an array of pixels.
    :param coordinates (numpy array):
    :return returns 8 arrays of coordinate pixels:
    '''
    x = np.zeros([len(coordinates), 3, 3])
    hsi_im = image.hsi_image[:, :, 0]
    hsi_im_border = np.pad(hsi_im, (1, 1), 'reflect')
    for i in xrange(-1, 2):
        for j in xrange(-1, 2):
            x[:, i + 1, j + 1] = directed_hue_color_distance(
                utils.evaluate_image(coordinates, hsi_im, outside_value=0.),
                utils.evaluate_image(coordinates + [i + 1, j + 1], hsi_im_border, outside_value=0.)
            )
    dx = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    dy = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    derivative_x = np.array([sum(sum(np.transpose(x * dx)))])
    derivative_y = np.array([sum(sum(np.transpose(x * dy)))])
    derivative = np.concatenate((derivative_x, derivative_y))
    return derivative
