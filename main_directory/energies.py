__author__ = 'jeroni'
import utils
import ctypes_utils as ctypes
import numpy as np


'''

                        LEARNING RATE

'''


def first_step_alpha(beta, curr_cage, grad_k):
    step = 0.001
    alpha = 0.05
    while all(multiple_norm(alpha * grad_k) <= beta):
        alpha += step
    return alpha


def second_step_alpha(alpha, curr_cage, grad_k, band_size, affine_contour_coord, contour_size, current_energy, image,
                      constraint_params):
    d, k = constraint_params
    step = 0.2
    next_energy = current_energy + 1
    alpha += step
    nrow, ncol = image.shape
    while current_energy < next_energy:
        alpha -= step

        # calculate new contour_coord
        contour_coord = np.dot(affine_contour_coord, curr_cage - grad_k * alpha)

        # Calculate new omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
        omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = ctypes.get_omega_1_and_2_coord(band_size,
                                                                                                  contour_coord,
                                                                                                  contour_size, ncol,
                                                                                                  nrow)

        affine_omega_1_coord, affine_omega_2_coord = ctypes.get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                           omega_2_coord, omega_2_size,
                                                                                           len(curr_cage),
                                                                                           curr_cage - grad_k * alpha)

        next_energy = mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                  image) + energy_constraint(curr_cage - grad_k * alpha, d, k)
    if alpha < 0.1:
        return 0
    return 1


'''

                        MEAN ENERGY

'''


def mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    omega_1 = mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
    omega_2 = mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
    energy = (omega_1 + omega_2) / float(2)
    return energy


def mean_energy_grad(omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    # Calculate Image gradient
    image_gradient = np.array(np.gradient(image))

    # Calculate Energy:
    omega_1 = mean_energy_grad_per_region(omega1_coord, affine_omega_1_coord, image, image_gradient)
    omega_2 = mean_energy_grad_per_region(omega2_coord, affine_omega_2_coord, image, image_gradient)
    energy = omega_1 + omega_2
    return energy


def mean_energy_per_region(omega_coord, affine_omega_coord, image):
    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    return np.dot(aux, np.transpose(aux))


def mean_energy_grad_per_region(omega_coord, affine_omega_coord, image, image_gradient):
    # E_mean
    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    image_gradient_by_point = [utils.evaluate_image(omega_coord, image_gradient[0], 0),
                               utils.evaluate_image(omega_coord, image_gradient[1], 0)]
    mean_derivative = np.dot(image_gradient_by_point, affine_omega_coord) / float(len(omega_coord))
    grad = gradient_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point, mean_derivative)
    return grad  # *(1/pow(omega_std, 2)) for GAUSS


def gradient_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point, mean_derivative):
    # image_gradient_by_point = np.transpose(image_gradient_by_point)
    aux_x = np.multiply(np.transpose(affine_omega_coord), image_gradient_by_point[0])
    aux_y = np.multiply(np.transpose(affine_omega_coord), image_gradient_by_point[1])
    aux_x = np.dot(aux, np.transpose(aux_x) - mean_derivative[0])
    aux_y = np.dot(aux, np.transpose(aux_y) - mean_derivative[1])
    aux_1 = np.transpose([aux_x, aux_y])
    return aux_1


def get_omega_mean(omega_coord, image):
    omega_boolean = utils.are_inside_image(omega_coord, image.shape)
    omega_coord_aux = omega_coord[omega_boolean]
    omega_intensity = sum(image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]])
    omega_mean = omega_intensity / (len(omega_boolean[omega_boolean]))
    omega_std = np.std(image[[omega_coord_aux[:, 0].tolist(), omega_coord_aux[:, 1].tolist()]])
    return omega_mean, omega_std


'''
                        Multi Dimensional MEAN ENERGY
'''


def mean_energy_multi(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image, type, weight):
    energy =
    for t,w in zip(type, weight):
        omega_1 = mean_energy_per_region(omega_1_coord, affine_omega_1_coord, image, )
        omega_2 = mean_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
        energy = (omega_1 + omega_2) / float(2)

    return energy


def mean_energy_per_region_multi(omega_coord, affine_omega_coord, image):
    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    return np.dot(aux, np.transpose(aux))


'''
                        GAUSSIAN ENERGY
'''


def gauss_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    '''
    Computes the Gaussian Energy of an Image
    :param omega_1_coord (numpy array): Omega coordinates for region Omega 1
    :param omega_2_coord (numpy array): Omega coordinates for region Omega 2
    :param affine_omega_1_coord (numpy array): Affine coordinates for region Omega 1
    :param affine_omega_2_coord (numpy array): Affine coordinates for region Omega 2
    :param image (numpy array): The Image
    :return:
    '''
    omega_1 = gauss_energy_per_region(omega_1_coord, affine_omega_1_coord, image)
    omega_2 = gauss_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
    energy = (omega_1 + omega_2) / float(2)
    return energy


def grad_gauss_energy(omega1_coord, omega2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
    '''
    Computes the derivative of the Gaussian Energy of an Image with respect to the control points
    :param omega1_coord (numpy array): Omega coordinates for region Omega 1
    :param omega2_coord (numpy array): Omega coordinates for region Omega 2
    :param affine_omega_1_coord (numpy array): Affine coordinates for region Omega 1
    :param affine_omega_2_coord (numpy array): Affine coordinates for region Omega 2
    :param image (numpy array): The Image
    :return:
    '''
    # Calculate Image gradient
    image_gradient = np.array(np.gradient(image))

    # Calculate Energy Per region:
    omega_1 = grad_gauss_energy_per_region(omega1_coord, affine_omega_1_coord, image, image_gradient)
    omega_2 = grad_gauss_energy_per_region(omega2_coord, affine_omega_2_coord, image, image_gradient)

    energy = omega_1 + omega_2
    return energy


def gauss_energy_per_region(omega_coord, affine_omega_coord, image):
    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    return np.dot(aux, np.transpose(aux))


def grad_gauss_energy_per_region(omega_coord, affine_omega_coord, image, image_gradient):
    # E_mean

    omega_mean, omega_std = get_omega_mean(omega_coord, image)
    aux = utils.evaluate_image(omega_coord, image, omega_mean) - omega_mean
    image_gradient_by_point = [utils.evaluate_image(omega_coord, image_gradient[0], 0),
                               utils.evaluate_image(omega_coord, image_gradient[1], 0)]
    mean_derivative = np.dot(image_gradient_by_point, affine_omega_coord) / float(len(omega_coord))
    grad = gradient_energy_for_each_vertex(aux, affine_omega_coord, image_gradient_by_point, mean_derivative)
    return grad  # *(1/pow(omega_std, 2)) for GAUSS


'''

                        CONSTRAINT ENERGIES

'''


def energy_constraint(vertices, d, k):
    energy = vertex_constraint(vertices, d) + edge_constraint(vertices, d)
    return energy * k  # Give a weight k


def grad_energy_constraint(vertices, d, k):
    grad = grad_vertex_constraint(vertices, d) + grad_edge_constraint(vertices, d)
    return grad * k  # Give a weight k


'''
                        VERTEX Minimum distance CONSTRAINT ENERGY
'''


def grad_vertex_constraint(vertices, d):
    grad_norm = np.zeros(vertices.shape)
    for i, vi in enumerate(vertices):
        for j, vj in enumerate(vertices[i + 1:]):
            # Avoid dividing by 0 and where the function is not defined
            dist_vertices = np.linalg.norm(vi - vj)
            if dist_vertices >= d or not dist_vertices > 0:
                continue
            aux = 2 * (vi - vj) * (d - dist_vertices) / float(dist_vertices)
            grad_norm[i] += aux
            grad_norm[(j + i + 1) % len(vertices)] += -aux

    return grad_norm


def vertex_constraint(vertices, d):
    '''
    Finds the energy of the vertex_constraint, i.e. The closer points are to each other the higher the energy
    :param vertices (numpy array of 2 dim numpy arrays): numpy List of vertices
    :param d:
    :return:
    '''
    vertex_energy = 0
    for i, vi in enumerate(vertices):
        for j, vj in enumerate(vertices[i + 1:]):
            vertex_energy += np.power(d - np.linalg.norm(vi - vj), 2) if np.linalg.norm(vi - vj) < d else 0
    return vertex_energy


'''
                        LENGTH CONSTRAINT ENERGY
'''


def length_constraint_energy(vertices, d):
    energy = 0
    for i in xrange(len(vertices)):
        # Distance between two neighboring points
        dist_pair = np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)])

        energy += np.linalg.norm(dist_pair - d)

    return energy


def grad_length_constraint_energy(vertices, d):
    grad_norm = np.zeros([vertices.shape])
    for i in xrange(len(vertices)):
        # Distance between two neighboring points
        dist_pair = np.linalg.norm(vertices[i] - vertices[(i + 1) % len(vertices)])
        denom1 = np.linalg.norm(dist_pair - d) * dist_pair

    return grad_norm


'''
                        EDGE CONSTRAINT ENERGY
'''


def grad_point_to_edge_energy_1(v, v_1, v_2, d):
    q = v_2 - v_1
    q_orth = perpendicular_vector(q)
    r = v - v_1
    grad = q_orth * (np.dot(q_orth, r)) / float(np.linalg.norm(q_orth) * abs(np.dot(q_orth, r)))
    return grad


def dist_point_to_edge(v, v_1, v_2, d):
    '''
    Finds the distance between a point v and a segment v1v2.
    CAVEAT: if the point cannot be projected perpendicular to the segment, None is returned*.
    :param v a (2 dim numpy array): A point
    :param v_1 (2 dim numpy array): initial point of the segment
    :param v_2 (2 dim numpy array): final point of the segment
    :return: a distance from the point to an edge if it exist.
    '''
    q = v_2 - v_1
    q_orth = perpendicular_vector(q)
    r = v - v_1
    # Calculate the range where the band where the distance can be well defined*
    range_ = np.dot(q, r) / np.power(np.linalg.norm(q), 2)
    if range_ < 0 or range_ > 1:
        return d
    else:
        distance = abs(np.dot(q_orth, r)) / np.linalg.norm(q_orth)
        return distance if distance < d else d


def point_to_edge_energy(v, v_1, v_2, d):
    '''
    Calculates the edge constraint energy of a point with regards to an edge.
    :param v (2 dim numpy array): A point
    :param v_1 (2 dim numpy array): initial point of the segment
    :param v_2 (2 dim numpy array): final point of the segment
    :param d: a scalar from which if the distance is too large, the energy is 0.
    :return: an energy value of the point v.
    '''
    energy = np.power(d - dist_point_to_edge(v, v_1, v_2, d), 2)
    return energy


def edge_constraint(vertices, d):
    num_points = len(vertices)
    edge_energy = 0
    for i, v in enumerate(vertices):
        for j in range(1, num_points - 1):
            v_1 = vertices[(i + j) % num_points]
            v_2 = vertices[(i + j + 1) % num_points]
            edge_energy += point_to_edge_energy(v, v_1, v_2, d)
    return edge_energy


def grad_edge_constraint(vertices, d):
    num_points = len(vertices)
    grad_energy = np.zeros(list(vertices.shape))
    for i, v in enumerate(vertices):
        for j in range(1, num_points - 1):
            i1_ = (i + j) % num_points
            i2_ = (i + j + 1) % num_points
            v_1 = vertices[i1_]
            v_2 = vertices[i2_]
            aux = grad_point_to_edge_energy_1(v, v_1, v_2, d)
            grad_energy[i] += 2 * (d - dist_point_to_edge(v, v_1, v_2, d)) * aux
    return grad_energy


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
    print 'Color 2',
    grad_energy_2 = grad_mean_color_energy_per_region(omega_2_coord, affine_omega_2_coord, image)
    return grad_energy_1 + grad_energy_2


def grad_mean_color_energy_per_region(omega_coord, affine_omega_coord, image):
    mean = mean_color_in_region(omega_coord, image)
    print mean
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
    for i in xrange(-1, 2):
        for j in xrange(-1, 2):
            x[:, i + 1, j + 1] = directed_hue_color_distance(
                utils.evaluate_image(coordinates, image.hsi_image[:, :, 0], outside_value=0.),
                utils.evaluate_image(coordinates + [i, j], image.hsi_image[:, :, 0], outside_value=0.))
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


'''

                    STOP CRITERIA

'''


def cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
    '''
    Checks if vertices cannot evolve anymore
    :param grad_k_3:
    :param grad_k_2:
    :param grad_k_1:
    :param grad_k:
    :return:
    '''
    if not all(np.diagonal(np.dot(grad_k, np.transpose(grad_k_2))) > 0.00001):
        return False
    if not all(np.diagonal(np.dot(grad_k_1, np.transpose(grad_k_3))) > 0.00001):
        return False
    if not all(np.diagonal(np.dot(grad_k, np.transpose(grad_k_1))) < 0.00001):
        return False
    return True


def perpendicular_vector(v):
    r""" Finds an arbitrary perpendicular vector to *v*."""
    # for two vectors (x, y) and (a, b) to be perpendicular,
    # the following equation has to be fulfilled
    # 0 = ax + by
    # x = y = 0 is not an acceptable solution
    if v[0] == v[1] == 0:
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array([v[1], 0])
    if v[1] == 0:
        return np.array([0, -v[0]])

    # The rotation matrix R is
    # 0  1
    # -1  0
    # so we have that Rv is:
    return np.array([v[1], -v[0]])


def multiple_project_gradient_on_axis(a, b):
    '''
    Finds a's projection on b
    :param a:
    :param b:
    :return:
    '''
    return np.transpose((multiple_dot_products(a, b) / np.power(multiple_norm(b), 2)) * np.transpose(b))


def normalize_vectors(vectors):
    '''
    Normalizes vectors
    :param vect:
    :return:
    '''
    vectors_aux = np.array([x / np.linalg.norm(x) for x in vectors])
    return vectors_aux


'''
    THE FOLLOWING FUNCTIONS ARE MADE TO ACCEPT MULTIPLE VECTORS IN THE FOLLOWING FORMAT:

    v1
    v2
    .
    .
    .
    vN
'''


def multiple_norm(a):
    '''

    :param a:
    :return:
    '''
    return np.array([np.linalg.norm(x) for x in a])


def multiple_normalize(a):
    '''

    :param a:
    :return:
    '''
    return np.array([x / np.linalg.norm(x) for x in a])


def multiple_standardize(a):
    '''

    :param a:
    :return:
    '''
    y = np.sqrt(sum(np.multiply(a, a).T))
    return a / y.mean()


def multiple_dot_products(a, b):
    c = a * b
    return np.array([sum(x) for x in c])

