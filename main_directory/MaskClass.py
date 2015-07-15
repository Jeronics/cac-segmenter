import numpy as np
from PIL import Image
import scipy
from scipy import misc
from matplotlib import pyplot as plt


class MaskClass:
    def __init__(self, mask=np.array([]), filename='', threshold=125., center=None, radius_point=None):
        self.mask = self.binarize_image(mask, threshold)
        self.height = mask.shape[0]
        self.width = mask.shape[1] if len(mask.shape) > 1 else None
        self.shape = mask.shape
        self.path = filename
        self.name = filename.split("/")[-1]
        self.spec_name = self.name.split('.png')[0]
        self.root = "/".join(filename.split("/")[:-1]) + "/"
        self.save_name = self.spec_name + '_out.png'
        self.center = center
        self.radius_point = radius_point

    def read_png(self, filename, threshold=125.):
        '''
        Return mask data from a raw PNG file as numpy array.
        :param name: a directory path to the png image.
        :return: An image matrix in the type (y,x)
        '''
        # mask = scipy.misc.imread(filename)
        # mask = mask.astype(np.float64)
        # self.__init__(mask, filename, threshold)

        im = Image.open(filename)  # scipy.misc.imread(name)
        im = im.convert('RGB').convert('L')
        im = np.array(im)
        im = im.astype(np.float64)
        self.__init__(im, filename=filename, threshold=threshold)

    def from_points_and_image(self, c, p, image):
        '''
        This function creates a mask and a sequence of cages.
        :param c:
        :param p:
        :param im_shape:
        :return:
        '''
        im_shape = image.shape
        radius = np.linalg.norm(np.array(c) - np.array(p))
        im = np.zeros(im_shape)
        mask_points = []

        # careful im_shape is (max(y), max(x))
        for y in xrange(im_shape[0]):
            for x in xrange(im_shape[1]):
                if pow(x - c[0], 2) + pow(y - c[1], 2) <= pow(radius, 2):
                    im[y, x] = 255
                    mask_points.append([y, x])
        im = np.array(im)
        im = im.astype(np.float64)
        self.__init__(im, filename='', threshold=125., center=c, radius_point=p)


    def plot_image(self, show_plot=True):
        im_aux = self.mask.astype('uint8')
        plt.gray()
        plt.imshow(im_aux, interpolation='nearest')
        plt.axis('off')
        if show_plot:
            plt.show()

    def binarize_image(self, image, threshold):
        im = image.copy()
        if len(im.shape) == 3:
            im = im[:, :, 0]
        for i in xrange(0, im.shape[0]):
            for j in xrange(0, im.shape[1]):
                if im[i, j] >= threshold:
                    im[i, j] = 255.
                else:
                    im[i, j] = 0.
        return im

    def reshape(self, new_width=-1, new_height=-1):
        height, width = self.shape
        im = Image.open(self.path)

        if not (new_width == -1 and new_height == -1):
            if new_width == -1:
                new_width = int(width * new_height / float(height))
            elif new_height == -1:
                new_height = int(height * new_width / float(width))

            reshaped_image = im.resize((new_width, new_height), Image.ANTIALIAS)
            self.__init__(np.array(reshaped_image), self.path)


    def save_image(self, filename=''):
        if filename == '':
            filename = self.save_path
        # Transpose the image before returning it to be saved
        mask = np.transpose(self.mask)
        scipy.misc.imsave(filename, mask)


