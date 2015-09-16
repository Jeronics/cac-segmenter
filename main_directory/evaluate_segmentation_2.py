__author__ = 'jeroni'
import os

print os.getcwd()
from MaskClass import MaskClass
from ImageClass import ImageClass
from CageClass import CageClass
import pandas as pd
import copy
import morphing
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_masks(filename):
    input_info = pd.read_csv(filename, sep='\t')
    return input_info


def return_contour(mask_obj):
    im_gray = mask_obj.mask.astype(np.uint8)
    ret, thresh = cv2.threshold(im_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def plot_contour_on_image(image_obj, mask_obj, title_name=''):
    contours , hierarchy = return_contour(mask_obj)
    image_obj.plot_image(show_plot=False, title_name=title_name)
    contour_index=np.argmax([len(contours[i]) for i in xrange(len(contours))])
    lon = contours[contour_index][:].T[0][0]
    lat = contours[contour_index][:].T[1][0]
    print contours
    plt.fill(lon, lat, fill=False, color='b', linewidth=2)
    plt.show()


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
        if i < 9 or i in [42, 86, 98]:
            continue
        image_obj, mask_obj, cage_obj, gt_mask = _load_model(x, params)
        results_cage_file = results_cage_folder + image_obj.spec_name + '.txt'
        if not os.path.exists(results_cage_file):
            continue
        results_cage = CageClass()
        results_cage.read_txt(results_cage_file)
        gt_mask.plot_image()
        morphed_mask = copy.deepcopy(mask_obj.mask)
        morphed_mask = morphed_mask * 0 + 255.
        resulting_mask = MaskClass()
        resulting_mask.mask = morphed_mask
        morphed_mask_final = morphing.morphing_mask(mask_obj, cage_obj, resulting_mask, results_cage)
        morphed_mask_final.plot_image()
        plot_contour_on_image(image_obj, morphed_mask_final, title_name=image_obj.spec_name)

if __name__ == '__main__':
    dataset = load_masks('AlpertGBB07_input.txt')
    results_folder = 'segment_results_alpert_3/' + 'MultiMixtureGaussianCAC/'
    parameter_file = results_folder + 'parameters.p'
    params = pickle.load(open(parameter_file, "rb"))
    walk_through_images(dataset, params, results_folder)