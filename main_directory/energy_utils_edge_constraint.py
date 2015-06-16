import numpy as np

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