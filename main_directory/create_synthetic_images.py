import sys
import utils
import numpy as np

image_matrix = np.zeros([300, 400, 3])

image_matrix[:, :] = [255., 0., 0.]
image_matrix[100:200, :] = [127.5, 255., 0.]
image_matrix[:, :100] = [255., 0., 0.]
image_matrix[:, 200:] = [255., 0., 0.]

Image = utils.ImageClass(im=image_matrix)

utils.mkdir('images_tester/synthetic_images/synthetic_image_2')

Image.save_image(filename='images_tester/synthetic_images/synthetic_image2/synthetic_2.png')

if __name__ == '__main__':
    print 'hi'