__author__ = 'jeronicarandellsaladich'
import numpy as np

from CACSegmenter import CACSegmenter
from MultiMixtureGaussianCAC import MultiMixtureGaussianCAC


def run_code(num_points, ratio, sigma, smallest_number, name=''):
    multi_mixture_gaussian_gray_cac = CACSegmenter(MultiMixtureGaussianCAC)
    new_parameters = {
        'num_points': [num_points],
        'ratio': [ratio],
        'smallest_number': [np.exp(smallest_number)]
    }
    multi_mixture_gaussian_gray_cac.parameters = new_parameters
    multi_mixture_gaussian_gray_cac.sigma = sigma
    parameter_list = multi_mixture_gaussian_gray_cac.get_parameters()
    dataset = multi_mixture_gaussian_gray_cac.load_dataset('AlpertGBB07_input_subtest.txt')
    results_folder = 'segment_subtests/' + multi_mixture_gaussian_gray_cac.CAC.__name__ + name + '/'
    print results_folder
    multi_mixture_gaussian_gray_cac.test_model(dataset, parameter_list[0], results_folder, plot_evolution=False)


def no_decimals(value):
    return ''.join(str(value).split('.'))


def no_sign(value):
    return ''.join(str(abs(value)).split('-'))


if __name__ == '__main__':
    num_points = [12, 16, 20]
    ratios = [1.05, 1.1]
    sigmas = [0.25, 0.5, 0.75]
    smallest_numbers = [-100, -200, -300]
    count = 0
    for p in num_points:
        if p == 12 or p == 16:
            continue
        for r in ratios:
            if r == 1.05:
                continue
            for s in sigmas:
                if s == 0.25:
                    continue
                for n in smallest_numbers:
                    if s==0.5 and n == -100:
                        continue
                    name = '_' + str(p) + '_' + no_decimals(r) + '_' + no_decimals(s) + '_' + no_sign(n)
                    run_code(p, r, s, n, name=name)
                    print name
                    count += 1

    print count



