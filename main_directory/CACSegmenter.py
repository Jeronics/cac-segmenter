from ctypes_utils import *
import time
import utils as utils
import energies


class CACSegmenter():
    def __init__(self, dataset_filename=None):
        self.dataset = self.load_dataset(dataset_filename)
        self.band_size = 500
        self.k = 50
        self.d = 10
        self.other = 10
        self.parameters = {
            'num_points': [6, 8, 10, 12, 14],
            'ratio': [1.05, 1.1, 1.15, 1.2, 1.25]
        }

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
        params_indexes={}
        specific_params = {}
        len(self.parameters)
        for params in self.parameters:
            len(self.parameters[params])
                specific_params[params] = p


    def load_dataset(self):
        return dataset

    def _partition_dataset(self, i):
        return train, test


class MeanColorCAC(CACSegmenter):
    def __init__(self, image_txt, center_point, outside_point, ratio):
        CACSegmenter.__init__(self)
        self.energy = 1
        image = utils.ImageClass()
        self.image_obj = image.read_png(filename=image_txt)
        mask = utils.MaskClass()
        self.mask_obj = mask.create_from_points(self.center_point, self.outside_point, self.ratio)


if __name__ == '__main__':
    color = MeanColorCAC()
    color.test_model()
    print color.other