import energies
from ctypes_utils import *
from utils import *
import energy_utils_cage_constraints as cage_constraint
import ctypes_utils as ctypes

class CAC():
    def __init__(self, image_obj, mask_obj, cage_obj):
        self.image_obj = image_obj
        self.mask_obj = mask_obj
        self.cage_obj = cage_obj



    def energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
        return None

    def energy_gradient(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
        return None

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=color,
                           points2=cage_obj.cage - alpha * 10 * grad_k)

    def segment(self, image_obj, mask_obj, _cage_obj, curr_cage_file, model='mean_model', plot_evolution=False):
        cage_obj = copy.deepcopy(self.cage_obj)
        if cage_out_of_the_picture(self.cage_obj.cage, self.image_obj.shape):
            print 'Cage is out of the image! Not converged. Try a smaller cage'
            return None

        contour_coord, contour_size = get_contour(self.mask_obj)
        affine_contour_coordinates = get_affine_contour_coordinates(contour_coord, cage_obj.cage)

        if plot_evolution:
            plotContourOnImage(contour_coord, self.image_obj.image, points=cage_obj.cage, color=[0., 0., 255.])

        # Update Step of contour coordinates
        contour_coord = np.dot(affine_contour_coordinates, cage_obj.cage)

        # copy of cage_obj
        iter = 0
        max_iter = 50
        max_iter_step_2 = 10
        first_stage = True
        grad_k_3, grad_k_2, grad_k_1, grad_k = np.zeros([cage_obj.num_points, 2]), np.zeros(
            [cage_obj.num_points, 2]), np.zeros(
            [cage_obj.num_points, 2]), np.zeros([cage_obj.num_points, 2])
        mid_point = sum(cage_obj.cage, 0) / cage_obj.num_points

        # PARAMETERS #
        # pixel steps
        beta = 5

        # Omega1 band size
        band_size = 80

        # Constraint Energy parameters
        # constraint energy. k=0 is none.
        k = 50

        # Algorithm requires k>=2*beta to work.
        d = 2 * beta
        constraint_params = [d, k]
        continue_while = True
        while continue_while:
            if iter > max_iter:
                continue_while = False
                print 'Maximum iterations reached'

            if cage_out_of_the_picture(cage_obj.cage, image_obj.shape):
                print 'Cage is out of the image! Not converged. Try a smaller cage'
                return None
            omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = get_omega_1_and_2_coord(band_size, contour_coord,
                                                                                               contour_size,
                                                                                               self.mask_obj.width,
                                                                                               self.mask_obj.height)

            affine_omega_1_coord, affine_omega_2_coord = get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                        omega_2_coord, omega_2_size,
                                                                                        self.cage_obj.num_points,
                                                                                        self.cage_obj.cage)

            # Update gradients
            grad_k_3 = grad_k_2.copy()
            grad_k_2 = grad_k_1.copy()
            grad_k_1 = grad_k.copy()
            grad_k = self.energy_gradient(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                          self.image_obj) + cage_constraint.grad_energy_constraint(cage_obj.cage, d, k)
            grad_k = energies.multiple_standardize(grad_k)
            if first_stage:
                mid_point = sum(cage_obj.cage, 0) / float(cage_obj.num_points)
                axis = mid_point - cage_obj.cage
                axis = energies.multiple_standardize(axis)
                grad_k = energies.multiple_project_gradient_on_axis(grad_k, axis)
                alpha = beta

            else:
                energy = self.energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                     self.image_obj) + cage_constraint.energy_constraint(self.cage_obj.cage, d, k)
                alpha_new = self.second_step_alpha(alpha, self.cage_obj.cage, grad_k, band_size,
                                                   affine_contour_coordinates,
                                                   contour_size, energy, self.image_obj, constraint_params)
                if alpha_new == 0:
                    continue_while = False
                    print 'Local minimum reached. no better alpha'
                    # return curr_cage_file

            # Calculate alpha
            # grad_k = normalize_vectors(grad_k)
            # print grad_k[2], curr_cage_file[2]
            # print curr_cage_file[-2], grad_k[-2]
            alpha = beta  # find_optimal_alpha(beta, curr_cage_file, grad_k)

            # if iter % 20 == 0:
            # plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=[0., 0., 255.],
            # points2=cage_obj.cage - alpha * 10 * grad_k)

            if plot_evolution:
                self._plotContourOnImage(contour_coord, self.image_obj, self.cage_obj, alpha, grad_k,
                                         color=[0., 0., 255.])

            # Update File current cage
            cage_obj.cage += - alpha * grad_k
            if first_stage and energies.cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
                first_stage = False
                print 'First stage reached'

            # Update contour coordinates
            contour_coord = np.dot(affine_contour_coordinates, self.cage_obj.cage)
            iter += 1

        if plot_evolution:
            self._plotContourOnImage(contour_coord, self.image_obj, self.cage_obj, 0, grad_k)
        return cage_obj


    def second_step_alpha(self, alpha, curr_cage, grad_k, band_size, affine_contour_coord, contour_size, current_energy,
                          image_obj, constraint_params):

        d, k = constraint_params
        step = 0.2
        next_energy = current_energy + 1
        alpha += step
        nrow, ncol = self.image_obj.shape
        while current_energy < next_energy:
            alpha -= step

            # calculate new contour_coord
            contour_coord = np.dot(affine_contour_coord, curr_cage - grad_k * alpha)

            # Calculate new omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
            omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = ctypes.get_omega_1_and_2_coord(band_size,
                                                                                                      contour_coord,
                                                                                                      contour_size,
                                                                                                      ncol, nrow)

            affine_omega_1_coord, affine_omega_2_coord = ctypes.get_omega_1_and_2_affine_coord(omega_1_coord,
                                                                                               omega_1_size,
                                                                                               omega_2_coord,
                                                                                               omega_2_size,
                                                                                               len(curr_cage),
                                                                                               curr_cage - grad_k * alpha)

            next_energy = self.energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                      self.image_obj) + cage_constraint.energy_constraint(curr_cage - grad_k * alpha, d,
                                                                                          k)
        if alpha < 0.1:
            return 0
        return 1