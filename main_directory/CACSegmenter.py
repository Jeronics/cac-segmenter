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
        assert os.path.isfile(dataset_name)
        dataset = pd.read_csv(dataset_name, sep='\t')
        return dataset

    def _partition_dataset(self, dataset_name, k=):
        dataset =
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


    def train_model(self, input_file, CV=5):
        '''
        This function uses cross validation to lean the optimal parameters
        :return:
        '''
        specific_params = list(ParameterGrid(self.parameters))

        print specific_params


