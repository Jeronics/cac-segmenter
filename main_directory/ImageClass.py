import numpy as np
from PIL import Image
import energy_utils_mean_hue as hue_mean
import rose_graph
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


class ImageClass:
    def __init__(self, im=np.array([]), filename=''):

        # FROM RGBA to RGB if necessary
        if im.shape == 2 and im.shape[2] == 4:
            self.image = im[:, :, 0:3]
        else:
            self.image = im
        # Gray image
        self.gray_image = self.rgb2gray(self.image)
        # Hue image
        self.hsi_image = self.rgb2hsi(self.image)
        self.shape = im.shape[:2]
        self.height = im.shape[0]
        self.width = im.shape[1] if len(im.shape) > 1 else None
        self.path = filename
        self.name = filename.split("/")[-1]
        self.spec_name = self.name.split('.png')[0]
        self.root = "/".join(filename.split("/")[:-1]) + "/"
        self.save_name = self.spec_name + '_out.png'

    def read_png(self, filename):
        '''
        Return image data from a raw PNG file as numpy array.
        :param name: a directory path to the png image.
        :return: An image matrix in the type (y,x)
        '''
        # im = scipy.misc.imread(filename)
        # im = im.astype(np.float64)
        # self.__init__(im, filename)
        im = Image.open(filename)
        im = im.convert('RGB')
        im = np.array(im)
        im = im.astype(np.float64)
        self.__init__(im, filename)

    def plot_image(self, show_plot=True):
        im_aux = self.image.astype('uint8')
        plt.gray()
        plt.imshow(im_aux, interpolation='nearest')
        plt.axis('off')
        if show_plot:
            plt.show()

    def plot_hsi_image(self, show_plot=True):

        plt.subplot(221)
        plt.gray()
        plt.imshow(self.hsi_image[:, :, 0] / (2 * 3.14) * 255., interpolation='nearest')
        plt.axis('off')

        plt.subplot(222, projection='polar')
        plt.gray()

        azi = self.hsi_image[:, :, 0] / (2 * 3.14) * 360.
        azi = azi.flatten()
        azi = list(azi)
        azi.append(359.)
        azi = np.array(azi)
        z = np.cos(np.radians(azi / 2.))
        coll = rose_graph.rose(azi, z=z, bidirectional=False, bins=50)
        plt.xticks(np.radians(range(0, 360, 10)),
                   ['Red', '', '', 'Red-Magenta', '', '', 'Magenta', '', '', 'Magenta-Blue', '', '', 'Blue', '', '',
                    'Blue-Cyan', '', '', 'Cyan', '', '', 'Cyan-Green', '', '', 'Green', '', '', 'Green-Yellow', '', '',
                    'Yellow', '', '', 'Yellow-Red', '', ''])
        plt.colorbar(coll, orientation='horizontal')
        plt.xlabel('A rose diagram colored by a second variable')
        plt.rgrids(range(5, 20, 5), angle=360)

        plt.subplot(223)
        plt.imshow(self.hsi_image[:, :, 0] / (2 * 3.14) * 255., cmap=matplotlib.cm.hsv)
        plt.axis('off')

        plt.subplot(224)
        data = self.hsi_image[:, :, 0] / (2 * 3.14) * 255.
        data = data.flatten()
        data = list(data)
        data.append(359.)
        n, bins, patches = plt.hist(data, 36)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # scale values to interval [0,pi]
        col = bin_centers
        col /= 360
        col = col
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', matplotlib.cm.hsv(c))
        plt.xticks([np.ceil(x) for i, x in enumerate(bins) if i % 3 == 0])
        if show_plot:
            plt.show()


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
        scipy.misc.imsave(filename, self.image)

    def rgb2gray(self, rgb):
        if len(rgb.shape) == 3:
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = rgb
        return gray

    def rgb2hsi(self, rgb):
        hsi = rgb.copy()
        if len(rgb.shape) == 3:
            coordinates = np.array([[i, j] for i in np.arange(rgb.shape[0]) for j in np.arange(rgb.shape[1])])
            hsi[coordinates[:, 0], coordinates[:, 1]] = hue_mean.rgb_to_hsi(coordinates, rgb)
        else:
            hsi = rgb
        return hsi