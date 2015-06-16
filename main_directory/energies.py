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

