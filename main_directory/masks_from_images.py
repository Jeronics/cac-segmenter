import utils
import os
# from pylab import close
from matplotlib.pyplot import ginput, close


def check_for_ground_truth(image_path, gt_file_name):
    if os.path.exists(gt_file_name):
        with open(gt_file_name, 'r') as f:
            gt_images = f.read().split('\n')
        for path in gt_images:
            if image_path.split("/")[-1].split('.')[0] in path.split("/")[-1].split('.')[0]:
                return path
    return False


def read_clicked_points(image_name):
    image = utils.ImageClass()
    image.read_png(image_name)
    print "Please click:", image_name
    image.plot_image(show_plot=False)
    input_points = ginput(2)
    print("clicked", input_points)
    close()
    center_point = [input_points[0][0], input_points[0][1]]
    radius_point = [input_points[1][0], input_points[1][1]]
    return center_point, radius_point


def create_dataset(images_file_name, gt_file_name, output_file_name, only_with_gt=False):
    # Read Images in the images_file_name
    with open(images_file_name, 'r') as f:
        input_images = f.read().split('\n')
    input_images = [im for im in input_images if im != ''][0:]
    # Open Input file to write in
    print input_images
    f = open(output_file_name, 'w')

    id = 0
    f.write(
        'image_name' + '\t'
        + 'center_x' + '\t'
        + 'center_y' + '\t'
        + 'radius_x' + '\t'
        + 'radius_y' + '\t'
        + 'gt_name' + '\n'
    )
    for image_name in input_images:
        gt = check_for_ground_truth(image_name, gt_file_name)
        if not gt and only_with_gt:
            continue
        center_point, radius_point = read_clicked_points(image_name)
        f.write(
            image_name + '\t'
            + str(center_point[0]) + '\t'
            + str(center_point[1]) + '\t'
            + str(radius_point[0]) + '\t'
            + str(radius_point[1])
        )
        if gt:
            f.write('\t' + gt + '\n')
        else:
            f.write('\t' + '0' + '\n')
        id += 1
    f.close()


if __name__ == '__main__':
    dataset_name = 'alpert'
    if dataset_name == 'bsds300':
        file_name = '../../BSDS300_images.txt'
        gt_file_name = '../../BSDS300_gt.txt'
        output_file_name = 'BSDS300_input.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'alpert':
        file_name = '../../AlpertGBB07_images.txt'
        gt_file_name = '../../AlpertGBB07_gt.txt'
        output_file_name = 'AlpertGBB07_input_2.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)
    if dataset_name == 'synthetic':
        file_name = '../../synthetic_images.txt'
        gt_file_name = '../../synthetic_gt.txt'
        output_file_name = 'synthetic_input.txt'
        create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=True)