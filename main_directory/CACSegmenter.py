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

    def energy(self):
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

    def _

    def _evaluate_model(self, dataset, parameters, CV=5):
        split_points = [int(i * len(dataset) / 5.) for i in xrange(CV + 1)]
        split_points[0] = -1
        split_points[-1] = len(dataset)

        print split_points
        for i in xrange(CV):
            Train, Test = self._partition_dataset(dataset, i, CV)

            self.test_model()
            for j in xrange(CV):
                learn, validate = self._partition_dataset(Train,j,CV)





        return

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

