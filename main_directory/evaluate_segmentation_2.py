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
    contours, hierarchy = return_contour(mask_obj)
    image_obj.plot_image(show_plot=False, title_name=title_name)
    contour_index = np.argmax([len(contours[i]) for i in xrange(len(contours))])
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
        if i < 0 or i in [42, 86, 98]:
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

good = ['0677845-r1-067-32_a', '100_0109', '100_0497', '112255696146', '113334665744', '20060319_087', 'aaa',
     'animal_5_bg_020803', 'b14pavel013', 'b19objects118', 'b1chesnuttame', 'b1snake001', 'b20nature_landscapes129',
     'b2chopper008', 'b2pods001',
     'b4nature_animals_land009''b7nature_trees002', 'bbmf_lancaster_july_06', 'beltaine_4_bg_050502', 'boy_float_lake',
     'bream_in_basin', 'broom07',
     'buggy_005', 'carrigafoyle_castle_ext', 'caterpiller', 'chaom38', 'cheeky_penguin', 'crw_0384', 'culzeancastle',
     'dsc_0959',
     'dsc_422910022', 'dsc01236', 'dsc01239_d', 'dsc04575', 'dscf0459', 'dscf3583', 'dscf3623', 'dscf3772', 'dscn0756',
     'dscn1908',
     'dscn2064', 'dscn2154', 'egret_face', 'estb_img_6461_', 'fullicewater', 'hot_air_balloons_05', 'hpim1242',
     'hpim5083_morguefile',
     'imagen_072__1_', 'img_1516', 'img_2528', 'img_2577', 'img_2592_f', 'img_3083_modif', 'img_3803', 'img_4730_modif',
     'ireland_62_bg_061502',
     'kconnors030466', 'leafpav', 'london_zoo3', 'mexico3', 'moth061304_0679', 'nitpix_p1280114', 'oscar2005_05_07',
     'outside_guggenheim_walls', 'pic0203vn0092', 'pic0203vn0105', 'pic106470172014', 'pic1080629574',
     'pic109250805856',
     'pict2605', 'postjp',
     'redberry_rb03', 'san_andres_130', 'sg_01_img_1943_tratada', 'sharp_image', 'skookumchuk_starfish1', 'snow2_004',
     'yokohm060409_dyjsn191']
bad = ['0677845-r1-067-32_a',
     '110016671724',
     '114591144943',
     'b7nature_trees002',
     'buggy_005',
     'bw4',
     'carriage',
     'carrigafoyle_castle_ext',
     'chain98',
     'chaom38',
     'cheeky_penguin',
     'dsc_0959',
     'dscf0034_l',
     'dscf3583',
     'dscn2064',
     'dscn6805',
     'egret_face',
     'estb_img_6461_',
     'europe_holiday_484',
     'img_1516',
     'img_2592_f',
     'img_4214',
     'img_7359_copia',
     'imgp2712',
     'ireland_62_bg_061502',
     'london_zoo3',
     'matsumt060102_dyj08',
     'osaka060102_dyjsn071',
     'outside_guggenheim_walls',
     'palovigna',
     'pic0203vn0092',
     'pic1092515922117',
     'pict2272',
     'picture_458',
     'san_andres_130',
     'sharp_image',
     'tendrils']