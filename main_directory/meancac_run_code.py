__author__ = 'jeronicarandellsaladich'
import numpy as np

from CACSegmenter import CACSegmenter
from MeanCAC import MeanCAC


def run_code(num_points, ratio, sigma, smallest_number, name=''):
    mean_cac = CACSegmenter(MeanCAC)
    new_parameters = {
        'num_points': [num_points],
        'ratio': [ratio],
    }
    mean_cac.parameters = new_parameters
    mean_cac.sigma = sigma
    parameter_list = mean_cac.get_parameters()
    dataset = mean_cac.load_dataset('AlpertGBB07_input_subtest.txt')
    results_folder = 'segment_alpert_mean_subtest/' + mean_cac.CAC.__name__ + name + '/'
    print results_folder
    mean_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=False)


def no_decimals(value):
    return ''.join(str(value).split('.'))


def no_sign(value):
    return ''.join(str(abs(value)).split('-'))


if __name__ == '__main__':
    num_points = [16, 20, 24]
    ratios = [1.05, 1.1]
    sigmas = [0.25, 0.5, 0.75]
    count = 0
    for p in num_points:
        for r in ratios:
            for s in sigmas:
                name = '_' + str(p) + '_' + no_decimals(r) + '_' + no_decimals(s)
                run_code(p, r, s, None, name=name)
                print name
                count += 1

    print count



