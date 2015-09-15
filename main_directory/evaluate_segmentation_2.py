__author__ = 'jeroni'
import os

print os.getcwd()
from MaskClass import MaskClass
from ImageClass import ImageClass
from CageClass import CageClass
import pandas as pd
import copy
import morphing


def load_masks(filename):
    input_info = pd.read_csv(filename, sep='\t')
    return input_info


def _load_model(x, parameters):
    image = ImageClass()
    image.read_png(x.image_name)
    mask = MaskClass()
    mask.from_points_and_image([x.center_x, x.center_y], [x.radius_x, x.radius_y], image)
    cage = CageClass()
    cage.create_from_points([x.center_x, x.center_y], [x.radius_x, x.radius_y], parameters['ratio'],
                            parameters['num_points'], filename='hello_test')
    gt_mask = MaskClass()
    if x.gt_name:
        gt_mask.read_png(x.gt_name)
    else:
        gt_mask = None
    return image, mask, cage, gt_mask


def walk_through_images(dataset, params, results_cage_folder):
    for i, x in dataset.iterrows():
        if i < 0 or i in [42, 86, 98]:
            continue
        image_obj, mask_obj, cage_obj, gt_mask = _load_model(x, params)
        results_cage_file = results_cage_folder + image_obj.spec_name + '.txt'
        results_cage = CageClass()
        results_cage.read_txt(results_cage_file)
        image_obj.plot_image()
        mask_obj.plot_image()
        gt_mask.plot_image()
        morphed_mask = copy.deepcopy(mask_obj.mask)
        morphed_mask = morphed_mask*0+255.
        resulting_mask = MaskClass()
        resulting_mask.mask = morphed_mask
        resulting_mask.plot_image()
        morphed_mask_final = morphing.morphing_mask(mask_obj, cage_obj, resulting_mask, results_cage)
        morphed_mask_final.plot_image()


if __name__ == '__main__':
    dataset = load_masks('AlpertGBB07_input.txt')
    params = {
        'num_points': 12,
        'ratio': 1.05
    }

    results_folder = 'segment_results_alpert_3/' + 'MultiMixtureGaussianCAC/'
    walk_through_images(dataset, params, results_folder)