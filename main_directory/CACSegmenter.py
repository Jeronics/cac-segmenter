from ctypes_utils import *
import time
import utils as utils
import energies
from sklearn.grid_search import ParameterGrid
import pandas as pd


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

        return resulting_cages, evaluation

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

    def _find_best_model(self, dataset, parameters, CV=5):
        parameters_performance = pd.DataFrame(self.get_parameters())
        for i in xrange(CV):
            _, Test = self._partition_dataset(dataset, i, CV)
            performance = []
            for p in self.get_parameters():
                performance.append(self.test_model(Test, p))
            # Add a column with the performance of the method on the dataset
            parameters_performance[str(i)] = performance
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

