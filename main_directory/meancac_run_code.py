__author__ = 'jeronicarandellsaladich'

from CACSegmenter import CACSegmenter
from OriginalGaussianCAC import OriginalGaussianCAC
from MixtureGaussianCAC import MixtureGaussianCAC


def run_code(num_points, ratio, sigma, smallest_number, name=''):
    original_gaussian = CACSegmenter(MixtureGaussianCAC)
    new_parameters = {
        'num_points': [num_points],
        'ratio': [ratio],
    }
    original_gaussian.parameters = new_parameters
    original_gaussian.sigma = sigma
    parameter_list = original_gaussian.get_parameters()
    dataset = original_gaussian.load_dataset('BSDS300_input.txt')
    results_folder = 'segment_bsds300_mixt_gauss/' + original_gaussian.CAC.__name__ + name + '/'
    print results_folder
    original_gaussian.test_model(dataset, parameter_list[0], results_folder, plot_evolution=False)


def no_decimals(value):
    return ''.join(str(value).split('.'))

def no_sign(value):
    return ''.join(str(abs(value)).split('-'))


if __name__ == '__main__':
    # num_points = [12]
    # ratios = [1.05]
    # sigmas = [1.25]
    # count = 0
    # for p in num_points:
    #     for r in ratios:
    #         for s in sigmas:
    #             name = '_' + str(p) + '_' + no_decimals(r) + '_' + no_decimals(s)
    #             run_code(p, r, s, None, name=name)
    #             print name
    #             count += 1
    #
    # print count
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


