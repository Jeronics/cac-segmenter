from ctypes_utils import *
import time
import utils as utils
import energies
from sklearn.grid_search import ParameterGrid


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
        return dataset_name

    def _partition_dataset(self, dataset_name):
        train = None
        test = None
        return train, test

    def energy(self):
        return None

    def _gradient_descent(self, images):
        return 0

    def test_model(self, image_obj, cage_obj, plot_evolution=False):
        '''
        segments a single image given a set of parameters.
        :return:
        '''
        return evaluation


    def train_model(self, Images, masks, C=5):
        '''
        This function uses cross validation to lean the optimal parameters
        :return:
        '''
        specific_params = list(ParameterGrid(self.parameters))

        print specific_params


class MeanColorCAC(CACSegmenter):
    def __init__(self):
        CACSegmenter.__init__(self)
        self.energy = 1
        self.parameters['other'] = [1, 2, 3, 4]


if __name__ == '__main__':
    color = MeanColorCAC()
    color.train_model(None, None)
    print color.other