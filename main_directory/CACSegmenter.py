from ctypes_utils import *
import time
import utils as utils
import energies
from sklearn.grid_search import ParameterGrid
import pandas as pd
from cac_segmenter import cac_segmenter
from ctypes_utils import *
import time
from utils import *
import energies



class CACSegmenter():
    def __init__(self):
        self.band_size = 500
        self.k = 50
        self.d = 10
        self.other = 10
        self.parameters = {
            'num_points': [6, 8, 10, 12, 14],
            'ratio': [1.05, 1.1, 1.15, 1.2, 1.25],
        }

    def _load_dataset(self, dataset_name):
        assert os.path.isfile(dataset_name), 'The input dataset file name is not valid!'
        dataset = pd.read_csv(dataset_name, sep='\t')
        return dataset

    def _energy(self):
        return None

    def _gradient_descent(self, images):
        return 0

    def test_model(self, dataset, params, plot_evolution=False):
        '''
        segments a group of image given a set of parameters. If the ground_truth exists it returns an evaluation
        :return resulting cages:
        '''
        for i, x in dataset.iterrows():
            print x.image_name
            image = utils.ImageClass()
            image.read_png(x.image_name)
            mask = utils.MaskClass()
            mask.
        return 0
        # return resulting_cages, evaluation

    def _partition_dataset(self, dataset, i_th, CV):
        '''
        Divides the dataset into Train or Test based on the i_th partition
        :param dataset (pandas dataframe):
        :param i_th (int):
        :param CV (int):
        :return: Train and Test pandas dataframes
        '''
        split_points = [int(i * len(dataset) / 5.) for i in xrange(CV + 1)]
        split_points[0] = -1
        split_points[-1] = len(dataset)
        a = split_points[i_th] + 1
        b = split_points[i_th + 1]
        Test = dataset[a:b]
        Train = pd.concat([dataset[:a], dataset[b:]])
        return Train, Test

    def _find_best_model(self, dataset, CV=5):
        parameters_performance = pd.DataFrame(self.get_parameters())
        performance_df = pd.DataFrame(dtype=float)
        for i in xrange(CV):
            _, Test = self._partition_dataset(dataset, i, CV)
            performance = []
            for p in self.get_parameters():
                performance.append(self.test_model(Test, p))
            # Add a column with the performance of the method on the dataset
            performance_df[str(i)] = performance
        parameters_performance['arithmetic_mean'] = performance_df.mean(axis=1)
        parameters_performance['harmonic_mean'] = len(performance_df.columns) / (1 / performance_df).sum(axis=1)
        parameters_performance = pd.concat([parameters_performance, performance_df], axis=1)
        return parameters_performance

    def _evaluate_method(self):
        return 0

    def get_parameters(self):
        '''
        Returns a list of all possible combinations of parameters.
        :return: list with a dictionary of parameter_name: value
        '''
        specific_params = list(ParameterGrid(self.parameters))
        return specific_params

    def train_model(self, input_file, CV=5):
        '''
        This function uses cross validation to lean the optimal parameters
        :return:
        '''
        dataset = self._load_dataset(input_file)
        parameter_performances = pd.DataFrame(self.get_parameters())
        self._cross_validation(dataset, CV)

    def cac_segmenter(self, image_obj, mask_obj, cage_obj, curr_cage_file, model='mean_model', plot_evolution=False):
        if cage_out_of_the_picture(cage_obj.cage, image_obj.shape):
            print 'Cage is out of the image! Not converged. Try a smaller cage'
            return None

        contour_coord, contour_size = get_contour(mask_obj)

        affine_contour_coordinates = get_affine_contour_coordinates(contour_coord, cage_obj.cage)

        if plot_evolution:
            plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=[0., 0., 255.])

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
                                                                                               contour_size, mask_obj.width,
                                                                                               mask_obj.height)

            affine_omega_1_coord, affine_omega_2_coord = get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                        omega_2_coord, omega_2_size,
                                                                                        cage_obj.num_points, cage_obj.cage)

            # Update gradients
            grad_k_3 = grad_k_2.copy()
            grad_k_2 = grad_k_1.copy()
            grad_k_1 = grad_k.copy()
            grad_k = energies.mean_energy_grad(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                               image_obj.gray_image) + energies.grad_energy_constraint(cage_obj.cage, d, k)
            grad_k = energies.multiple_normalize(grad_k)
            if first_stage:
                mid_point = sum(cage_obj.cage, 0) / float(cage_obj.num_points)
                axis = mid_point - cage_obj.cage
                axis = energies.multiple_normalize(axis)
                grad_k = energies.multiple_project_gradient_on_axis(grad_k, axis)
                alpha = beta

            else:
                energy = energies.mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                              image_obj.gray_image) + energies.energy_constraint(cage_obj.cage, d, k)
                alpha_new = energies.second_step_alpha(alpha, cage_obj.cage, grad_k, band_size, affine_contour_coordinates,
                                                       contour_size, energy, image_obj.gray_image, constraint_params)
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
                plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=[0., 0., 255.],
                                   points2=cage_obj.cage - alpha * 10 * grad_k)

            # Update File current cage
            cage_obj.cage += - alpha * grad_k
            if first_stage and energies.cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
                first_stage = False
                print 'First stage reached'

            # Update contour coordinates
            contour_coord = np.dot(affine_contour_coordinates, cage_obj.cage)
            iter += 1
        return cage_obj