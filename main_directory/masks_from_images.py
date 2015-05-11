import utils
import os
from pylab import ginput, close


def check_for_ground_truth(image_path, gt_file_name):
    if os.path.exists(gt_file_name):
        with open(gt_file_name, 'r') as f:
            gt_images = f.read().split('\n')
        for path in gt_images:
            if image_path.split("/")[-1] == path.split("/")[-1]:
                return path
    return False


def read_clicked_points(image_name):
    image = utils.ImageClass()
    image.read_png(image_name)
    print("Please click")
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

    # Open Input file to write in
    f = open(output_file_name, 'w')

    id = 0
    for image_name in input_images:
        gt = check_for_ground_truth(image_name, gt_file_name)
        if not gt and only_with_gt:
            continue
        center_point, radius_point = read_clicked_points(image_name)
        f.write(
            str(id) + '\t'
            + image_name + '\t'
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
    file_name = 'images.txt'
    gt_file_name = 'gt_images.txt'
    output_file_name = 'input.txt'
    create_dataset(file_name, gt_file_name, output_file_name, only_with_gt=False)