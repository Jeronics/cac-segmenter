__author__ = 'jeronicarandellsaladich'
import numpy as np

from CACSegmenter import CACSegmenter
from MeanCAC import MeanCAC
from HueMeanCAC import HueMeanCAC


def run_code(num_points, ratio, sigma, smallest_number, name=''):
    hue_mean = CACSegmenter(HueMeanCAC)
    new_parameters = {
        'num_points': [num_points],
        'ratio': [ratio],
    }
    hue_mean.parameters = new_parameters
    hue_mean.sigma = sigma
    parameter_list = hue_mean.get_parameters()
    dataset = hue_mean.load_dataset('AlpertGBB07_input_subtest.txt')
    results_folder = 'segment_alpert_hue_subtest/' + hue_mean.CAC.__name__ + name + '/'
    print results_folder
    hue_mean.test_model(dataset, parameter_list[0], results_folder, plot_evolution=False)


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
                # HueMeanCAC_24_105_025
                if not (p==24 and r==1.05 and s==0.25):
                    continue
                name = '_' + str(p) + '_' + no_decimals(r) + '_' + no_decimals(s)
                run_code(p, r, s, None, name=name)
                print name
                count += 1

    print count



