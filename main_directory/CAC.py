from copy import deepcopy

import energies
from ctypes_utils import *
from utils import *
import energy_utils_cage_constraints as cage_constraint
import ctypes_utils as ctypes


class CAC():
    def __init__(self, image_obj, mask_obj, cage_obj, ground_truth_obj, type=None, weight=None, band_size=500):
        self.image_obj = image_obj
        self.mask_obj = mask_obj
        self.cage_obj = cage_obj
        self.ground_truth_obj = ground_truth_obj
        self.band_size = band_size
        self.type = type
        self.weight = weight
        self.standardize = True

    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        return None

    def energy_gradient(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord):
        return None

    def _plotContourOnImage(self, contour_coord, current_cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        plotContourOnImage(contour_coord, self.image_obj.image, points=current_cage_obj.cage, color=color,
                           points2=current_cage_obj.cage - alpha * 10 * grad_k)

    def segment(self, plot_evolution=False):
        current_cage_obj = deepcopy(self.cage_obj)
        if cage_out_of_the_picture(current_cage_obj.cage, self.image_obj.shape):
            print 'Cage is out of the image! Not converged. Try a smaller cage'
            return None

        contour_coord, contour_size = get_contour(self.mask_obj)
        affine_contour_coordinates = get_affine_contour_coordinates(contour_coord, current_cage_obj.cage)

        if plot_evolution:
            plotContourOnImage(contour_coord, self.image_obj.image, points=current_cage_obj.cage, color=[0., 0., 255.])

        # Update Step of contour coordinates
        contour_coord = np.dot(affine_contour_coordinates, current_cage_obj.cage)

        # copy of cage_obj
        iter = 0
        max_iter = 100
        first_stage = True
        grad_k_3, grad_k_2, grad_k_1, grad_k = np.zeros([current_cage_obj.num_points, 2]), np.zeros(
            [current_cage_obj.num_points, 2]), np.zeros(
            [current_cage_obj.num_points, 2]), np.zeros([current_cage_obj.num_points, 2])
        mid_point = sum(current_cage_obj.cage, 0) / current_cage_obj.num_points

        # PARAMETERS #
        # pixel steps
        beta = 3

        # Omega1 band size
        band_size = 500

        # Constraint Energy parameters
        # constraint energy. k=0 is none.
        k = np.power(10, 50)

        # Algorithm requires k>=2*beta to work.
        d = 2 * beta
        constraint_params = [d, k]
        continue_while = True
        energy = np.inf
        while continue_while:
            if iter > max_iter:
                continue_while = False
                print 'Maximum iterations reached'

            if cage_out_of_the_picture(current_cage_obj.cage, self.image_obj.shape):
                print 'Cage is out of the image! Not converged. Try a smaller cage'
                return None

            omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = get_omega_1_and_2_coord(band_size, contour_coord,
                                                                                               contour_size,
                                                                                               self.mask_obj.width,
                                                                                               self.mask_obj.height)
            if not omega_1_size:
                print 'Contour has closed in or expanded.'
                return None
            affine_omega_1_coord, affine_omega_2_coord = get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                        omega_2_coord, omega_2_size,
                                                                                        current_cage_obj.num_points,
                                                                                        current_cage_obj.cage)

            # Update gradients
            grad_k_3 = grad_k_2.copy()
            grad_k_2 = grad_k_1.copy()
            grad_k_1 = grad_k.copy()
            grad_k = self.energy_gradient(omega_1_coord, omega_2_coord, affine_omega_1_coord,
                                          affine_omega_2_coord) + cage_constraint.grad_energy_constraint(
                current_cage_obj.cage, d, k)
            if self.standardize:
                grad_k = energies.multiple_standardize(grad_k)
                # Maximum cut if greater than one
                grad_k[abs(grad_k) > 1] /= abs(grad_k[abs(grad_k) > 1])
            else:
                grad_k = energies.multiple_normalize(grad_k)
            if first_stage:
                mid_point = sum(current_cage_obj.cage, 0) / float(current_cage_obj.num_points)
                axis = mid_point - current_cage_obj.cage
                if self.standardize:
                    axis = energies.multiple_standardize(axis)
                else:
                    axis = energies.multiple_normalize(axis)
                grad_k = energies.multiple_project_gradient_on_axis(grad_k, axis)
                alpha = beta
            else:
                energy = self.energy(omega_1_coord, omega_2_coord, affine_omega_1_coord,
                                     affine_omega_2_coord) + cage_constraint.energy_constraint(current_cage_obj.cage, d,
                                                                                               k)
                # if previous_energy:
                #     if energy > previous_energy:
                #         print 'second'
                #         continue_while = False


                alpha=beta
                alpha_new = self.second_step_alpha(alpha, current_cage_obj.cage, grad_k, band_size,
                                                   affine_contour_coordinates, contour_size, energy,
                                                   constraint_params)
                if not alpha_new:
                    if alpha_new == 0:
                        continue_while = False
                        print 'Local minimum reached. no better alpha'
                    else:
                        print 'Contour has closed in or expanded.'
                        return None
                alpha = alpha_new

            # Calculate alpha
            # grad_k = normalize_vectors(grad_k)
            # print grad_k[2], curr_cage_file[2]
            # print curr_cage_file[-2], grad_k[-2]
            # find_optimal_alpha(beta, curr_cage_file, grad_k)

            # if iter % 20 == 0:
            # plotContourOnImage(contour_coord, self.mask_obj.image, points=cage_obj.cage, color=[0., 0., 255.],
            # points2=cage_obj.cage - alpha * 10 * grad_k)

            if plot_evolution:
                self._plotContourOnImage(contour_coord, current_cage_obj, alpha, grad_k, color=[0., 0., 255.])

            # Update File current cage
            current_cage_obj.cage += - alpha * grad_k
            if first_stage and energies.cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
                first_stage = False
                print 'First stage reached'

            # Update contour coordinates
            contour_coord = np.dot(affine_contour_coordinates, current_cage_obj.cage)
            iter += 1

        if plot_evolution:
            self._plotContourOnImage(contour_coord, current_cage_obj, 0, grad_k)
        return current_cage_obj


    def second_step_alpha(self, alpha, curr_cage, grad_k, band_size, affine_contour_coord, contour_size, current_energy,
                          constraint_params):
        d, k = constraint_params
        step = 0.2
        next_energy = np.inf
        alpha += step
        nrow, ncol = self.mask_obj.shape
        while current_energy < next_energy:
            alpha -= step

            # calculate new contour_coord
            contour_coord = np.dot(affine_contour_coord, curr_cage - grad_k * alpha)

            # Calculate new omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
            omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = ctypes.get_omega_1_and_2_coord(band_size,
                                                                                                      contour_coord,
                                                                                                      contour_size,
                                                                                                      ncol, nrow)
            if not omega_1_size:
                return None

            affine_omega_1_coord, affine_omega_2_coord = ctypes.get_omega_1_and_2_affine_coord(omega_1_coord,
                                                                                               omega_1_size,
                                                                                               omega_2_coord,
                                                                                               omega_2_size,
                                                                                               len(curr_cage),
                                                                                               curr_cage - grad_k * alpha)

            next_energy = self.energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord)
            aux = cage_constraint.energy_constraint(curr_cage - grad_k * alpha, d, k)
            next_energy += aux
        if alpha < 0.01:
            return 0
        return alpha

        # def second_step_alpha(self, alpha, curr_cage, grad_k, band_size, affine_contour_coord, contour_size, current_energy, constraint_params):
        #
        # d, k = constraint_params
        # step = 0.2
        #     next_energy = current_energy + 1
        #     alpha += step
        #     nrow, ncol = self.image_obj.shape
        #     while current_energy < next_energy:
        #         alpha -= step
        #
        #         # calculate new contour_coord
        #         contour_coord = np.dot(affine_contour_coord, curr_cage - grad_k * alpha)
        #
        #         # Calculate new omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
        #         omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = ctypes.get_omega_1_and_2_coord(band_size,
        #                                                                                                   contour_coord,
        #                                                                                                   contour_size,
        #                                                                                                   ncol, nrow)
        #         if not omega_1_size:
        #             return None
        #         affine_omega_1_coord, affine_omega_2_coord = ctypes.get_omega_1_and_2_affine_coord(omega_1_coord,
        #                                                                                            omega_1_size,
        #                                                                                            omega_2_coord,
        #                                                                                            omega_2_size,
        #                                                                                            len(curr_cage),
        #                                                                                            curr_cage - grad_k * alpha)
        #
        #         next_energy = self.energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord) + \
        #                       cage_constraint.energy_constraint(curr_cage - grad_k * alpha, d, k)
        #     if alpha < 0.1:
        #         return 0
        #     return 1