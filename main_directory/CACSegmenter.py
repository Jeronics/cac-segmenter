from ctypes_utils import *
import utils as utils
from sklearn.grid_search import ParameterGrid
import pandas as pd
from cac_segmenter import cac_segmenter
from ctypes_utils import *
import time
from utils import *
import energies
from cac_segmenter import cac_segmenter
import ctypes_utils as ctypes


class CACSegmenter():
    def __init__(self):
        self.band_size = 500
        self.k = 50
        self.d = 10
        self.other = 10
        self.parameters = {
            'num_points': [12, 14, 16],
            'ratio': [1.05, 1.1, 1.15, 1.2, 1.25],
        }

    def _load_dataset(self, dataset_name):
        assert os.path.isfile(dataset_name), 'The input dataset file name is not valid!'
        dataset = pd.read_csv(dataset_name, sep='\t')
        return dataset

    def _energy(self):
        return None


    def evaluate_results(self, image, cage, mask, resulting_cage, gt_mask, results_file='results_cages'):
        utils.mkdir(results_file)
        result_file = results_file + "/" + cage.save_name
        if not resulting_cage:
            print 'No convergence reached for the cac-segmenter'
        else:
            resulting_cage.save_cage(result_file)
            res_fold = results_file + "/" + 'result' + cage.spec_name.split("cage_")[-1] + '.png'
            result_mask = utils.create_ground_truth(cage, resulting_cage, mask)
            if result_mask:
                result_mask.save_image(filename=res_fold)
            print res_fold
            if gt_mask:
                sorensen_dice_coeff = utils.sorensen_dice_coefficient(gt_mask, result_mask)
                print 'Sorensen-Dice coefficient', sorensen_dice_coefficient


    def _load_model(self, x, parameters):
        image = utils.ImageClass()
        image.read_png(x.image_name)
        mask = utils.MaskClass()
        mask.from_points_and_image([x.center_x, x.center_y], [x.radius_x, x.radius_y], image, parameters['num_points'],
                                   'hello_test')
        cage = utils.CageClass()
        cage.create_from_points([x.center_x, x.center_y], [x.radius_x, x.radius_y], parameters['ratio'],
                                parameters['num_points'], filename='hello_test')
        gt_mask = utils.MaskClass()
        if x.gt_name:
            gt_mask.read_png(x.gt_name)
        else:
            gt_mask = None
        return image, mask, cage, gt_mask

    def test_model(self, dataset, params, results_folder, plot_evolution=False):
        '''
        segments a group of image given a set of parameters. If the ground_truth exists it returns an evaluation
        :return resulting cages:
        '''
        utils.mkdir(results_folder)
        for i, x in dataset.iterrows():
            image_obj, mask_obj, cage_obj, gt_mask = self._load_model(x, params)
            result = self.cac_segmenter(image_obj, mask_obj, cage_obj, None, model='mean_model',
                                        plot_evolution=plot_evolution)
            if result:
                self.evaluate_results(image_obj, cage_obj, mask_obj, result, gt_mask)
                result.save_cage(results_folder + '/' + image_obj.spec_name + '.txt')

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
        results_folder = 'results/'
        parameters_performance = pd.DataFrame(self.get_parameters())
        performance_df = pd.DataFrame(dtype=float)
        for i in xrange(CV):
            _, Test = self._partition_dataset(dataset, i, CV)
            performance = []
            for i, p in enumerate(self.get_parameters()):
                results_folder_p = results_folder + 'params' + i
                performance.append(self.test_model(Test, p, results_folder_p))
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

    def mean_energy(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
        return None

    def mean_energy_grad(self, omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord, image):
        return None

    def _plotContourOnImage(self, contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.]):
        plotContourOnImage(contour_coord, image_obj.image, points=cage_obj.cage, color=color,
                           points2=cage_obj.cage - alpha * 10 * grad_k)

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
                                                                                               contour_size,
                                                                                               mask_obj.width,
                                                                                               mask_obj.height)

            affine_omega_1_coord, affine_omega_2_coord = get_omega_1_and_2_affine_coord(omega_1_coord, omega_1_size,
                                                                                        omega_2_coord, omega_2_size,
                                                                                        cage_obj.num_points,
                                                                                        cage_obj.cage)

            # Update gradients
            grad_k_3 = grad_k_2.copy()
            grad_k_2 = grad_k_1.copy()
            grad_k_1 = grad_k.copy()
            grad_k = self.mean_energy_grad(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                           image_obj) + energies.grad_energy_constraint(cage_obj.cage, d, k)
            grad_k = energies.multiple_standardize(grad_k)
            # print 'NORMALIZED'
            # print grad_k
            if first_stage:
                mid_point = sum(cage_obj.cage, 0) / float(cage_obj.num_points)
                axis = mid_point - cage_obj.cage
                axis = energies.multiple_standardize(axis)
                grad_k = energies.multiple_project_gradient_on_axis(grad_k, axis)
                alpha = beta

            else:
                energy = self.mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                          image_obj) + energies.energy_constraint(cage_obj.cage, d, k)
                alpha_new = self.second_step_alpha(alpha, cage_obj.cage, grad_k, band_size,
                                                   affine_contour_coordinates,
                                                   contour_size, energy, image_obj, constraint_params)
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
                self._plotContourOnImage(contour_coord, image_obj, cage_obj, alpha, grad_k, color=[0., 0., 255.])

            # Update File current cage
            cage_obj.cage += - alpha * grad_k
            if first_stage and energies.cage_vertex_do_not_evolve(grad_k_3, grad_k_2, grad_k_1, grad_k):
                first_stage = False
                print 'First stage reached'

            # Update contour coordinates
            contour_coord = np.dot(affine_contour_coordinates, cage_obj.cage)
            iter += 1

        if plot_evolution:
            self._plotContourOnImage(contour_coord, image_obj, cage_obj, 0, grad_k)
        return cage_obj


    def second_step_alpha(self, alpha, curr_cage, grad_k, band_size, affine_contour_coord, contour_size, current_energy,
                          image_obj, constraint_params):

        d, k = constraint_params
        step = 0.2
        next_energy = current_energy + 1
        alpha += step
        nrow, ncol = image_obj.shape
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

            next_energy = self.mean_energy(omega_1_coord, omega_2_coord, affine_omega_1_coord, affine_omega_2_coord,
                                           image_obj) + energies.energy_constraint(curr_cage - grad_k * alpha, d, k)
        if alpha < 0.1:
            return 0
        return 1
