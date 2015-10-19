import os
# from pylab import close
# from matplotlib.pyplot import ginput, close
from ImageClass import ImageClass
from MaskClass import MaskClass

def check_for_ground_truth(image_path, gt_file_name):
    if os.path.exists(gt_file_name):
        with open(gt_file_name, 'r') as f:
            gt_images = f.read().split('\n')
        for path in gt_images:
            if image_path.split("/")[-1].split('.')[0] in path.split("/")[-1].split('.')[0]:
                return path
    return False


def read_clicked_points(image_name, mask_name):
    mask = MaskClass()
    mask.read_png(mask_name)

    image = ImageClass()
    image.read_png(image_name)
    print "Please click:", image_name
    # image.plot_image(show_plot=True)

    mask.save_image(filename='../../../../MATLAB/creaseg/creaseg/creaseg/data/alpert_reduced_gt/'+image.spec_name+'.png')

    return None


def create_dataset(images_file_name, gt_file_name, output_file_name, only_with_gt=False):
    # Read Images in the images_file_name
    with open(images_file_name, 'r') as f:
        input_images = f.read().split('\n')
    input_images = [im for im in input_images if im != ''][0:]
    # Open Input file to write in
    print input_images
    id = 0

    for image_name in input_images:
        gt = check_for_ground_truth(image_name, gt_file_name)
        if not gt and only_with_gt:
            continue
        print gt_file_name
        read_clicked_points(image_name, gt)


if __name__ == '__main__':
    dataset_name = 'alpert'
    if dataset_name == 'morphing_fruits':
        file_name = '../../morphing_fruits.txt'
        gt_file_name = '../../morphing_fruits_gt.txt'
        output_file_name = 'morphing_fruits_input.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'morphing_cars':
        file_name = '../../morphing_cars.txt'
        gt_file_name = '../../morphing_cars_gt.txt'
        output_file_name = 'morphing_cars_input.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'bsds300':
        file_name = '../../BSDS300_images.txt'
        gt_file_name = '../../BSDS300_gt.txt'
        output_file_name = 'BSDS300_input.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'bsds300_expanded':
        file_name = '../../Berkley_images.txt'
        gt_file_name = '../../BSDS300_gt_expanded.txt'
        output_file_name = 'BSDS300_input_expanded.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'alpert':
        file_name = '../../AlpertGBB07_images.txt'
        gt_file_name = '../../AlpertGBB07_gt.txt'
        output_file_name = 'AlpertGBB07_input_2.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'synthetic':
        file_name = '../../synthetic_mixture_images.txt'
        gt_file_name = '../../synthetic_gt.txt'
        output_file_name = 'synthetic_brightness_input.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'mri_images':
        file_name = '../../mri_images.txt'
        gt_file_name = '../../mri_gt.txt'
        output_file_name = 'mri_input.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)