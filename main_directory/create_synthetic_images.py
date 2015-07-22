import sys
import utils
import numpy as np

image_matrix = np.zeros([300, 400, 3])
#
# image_matrix[:, :] = [255., 0., 0.]
# image_matrix[100:200, :] = [0, 255., 0.]
# image_matrix[:, :100] = [255., 0., 0.]
# image_matrix[:, 200:] = [255., 0., 0.]

image_matrix[:, :] = [0., 0., 0.]
image_matrix[100:200, :] = [255., 255., 255.]
image_matrix[:, :100] = [0., 0., 0.]
image_matrix[:, 200:] = [0., 0., 0.]

#
# image_matrix[:, :] = [255., 0., 0.]
# image_matrix[100:200, :] = [127.5, 255., 0.]
# image_matrix[:, :100] = [255., 0., 0.]
# image_matrix[:, 200:] = [255., 0., 0.]

Image = utils.ImageClass(im=image_matrix)

utils.mkdir('images_tester/gt_synthetic_images/')

Image.save_image(filename='images_tester/gt_synthetic_images/gt_synthetic_1.png')

if __name__ == '__main__':
    print 'hi'