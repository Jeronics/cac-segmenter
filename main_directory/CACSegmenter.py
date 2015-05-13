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

    def _partition_dataset(self, dataset_name, CV=5):
        dataset = None
        train = None
        test = None
        return train, test

    def energy(self):
        return None

    def _gradient_descent(self, images):
        return 0

    def test_model(self, image_obj, cage_obj, plot_evolution=False):
        '''
        segments a group of image given a set of parameters. If the ground_truth exists it returns an evaluation
        :return resulting cages:
        '''
        return resulting_cages, evaluation

    def _cross_validation(self, dataset, parameters, CV=5):
        split_points = [int(i * len(dataset) / 5.) for i in xrange(CV + 1)]
        split_points[0] = -1
        split_points[-1] = len(dataset)

        print split_points
        for i in xrange(CV):
            print 'hi', split_points[i] + 1, split_points[i + 1]
            # _partition_dataset(i,CV)

    def train_model(self, input_file, CV=5):
        '''
        This function uses cross validation to lean the optimal parameters
        :return:
        '''
        dataset = self._load_dataset(input_file)
        specific_params = list(ParameterGrid(self.parameters))
        parameter_performances = pd.DataFrame(specific_params)  # columns=['partition_'+ str(i) for i in xrange(CV)]
        self._cross_validation(dataset, CV)
        print specific_params
        print parameter_performances

