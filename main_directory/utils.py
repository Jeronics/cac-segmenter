import numpy as np
from scipy import *
import matplotlib
from time import sleep

matplotlib.use("Qt5Agg")
import scipy
from scipy import misc
import os
import sys
import energies
import energy_utils_mean_hue as hue_mean
import rose_graph

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PIL import Image


class CageClass:
    def __init__(self, cage=np.array([]), filename=''):
        self.cage = cage
        self.shape = cage.shape
        self.num_points = len(cage)
        self.path = filename
        self.name = filename.split("/")[-1]
        self.spec_name = self.name.split('.txt')[0]
        self.root = "/".join(filename.split("/")[:-1]) + "/"
        self.save_name = self.spec_name + '_out.txt'

    def read_txt(self, filename):
        cage = np.loadtxt(filename, float)
        # Rotate the cage to (y,x)
        rot = np.array([[0, 1], [1, 0]])
        cage = np.dot(cage, rot)
        self.__init__(cage, filename)

    def create_from_points(self, c, p, ratio, num_cage_points, filename=''):
        '''
        This function instantiates a cage.
            The cages are created clockwise from (x,y)=( c_x + r*R, c_y)
        '''
        radius = np.linalg.norm(np.array(c) - np.array(p))
        cage = []
        for i in xrange(0, num_cage_points):
            angle = 2 * i * np.pi / num_cage_points
            x, y = radius * ratio * np.sin(angle), radius * ratio * np.cos(angle)
            cage.append([y + c[1], x + c[0]])
        self.__init__(cage=np.array(cage), filename='')
        return cage

    def save_cage(self, filename):
        text_file = open(filename, "w")

        for x, y in self.cage:
            # Un-Rotate to (y,x)
            text_file.write("%.8e\t%.8e\n" % (y, x))


class MaskClass:
    def __init__(self, mask=np.array([]), filename='', threshold=125.):
        self.mask = self.binarize_image(mask, threshold)
        self.height = mask.shape[0]
        self.width = mask.shape[1] if len(mask.shape) > 1 else None
        self.shape = mask.shape
        self.path = filename
        self.name = filename.split("/")[-1]
        self.spec_name = self.name.split('.png')[0]
        self.root = "/".join(filename.split("/")[:-1]) + "/"
        self.save_name = self.spec_name + '_out.png'

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
        self.__init__(im, filename, threshold)

    def from_points_and_image(self, c, p, image, num_cage_points, filename):
        '''
        This function creates a mask and a sequence of cages.
        :param c:
        :param p:
        :param im_shape:
        :param num_cage_points:
        :return:
        '''
        im_shape = image.shape
        radius = np.linalg.norm(np.array(c) - np.array(p))
        im = np.zeros(im_shape)
        print 'Shape', im_shape
        mask_points = []

        # careful im_shape is (max(y), max(x))
        for y in xrange(im_shape[0]):
            for x in xrange(im_shape[1]):
                if pow(x - c[0], 2) + pow(y - c[1], 2) <= pow(radius, 2):
                    im[y, x] = 255
                    mask_points.append([y, x])
        im = np.array(im)
        im = im.astype(np.float64)
        self.__init__(im, filename='', threshold=125.)
        cage = []
        ratio = 1.05
        for i in xrange(0, num_cage_points):
            angle = 2 * i * np.pi / num_cage_points
            x, y = radius * ratio * np.sin(angle), radius * ratio * np.cos(angle)
            cage.append([y + c[1], x + c[0]])
        # plotContourOnImage(np.array(mask_points), image.image,
        # points=cage)
        return cage


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
        print bins
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


# ########## VISUALITON
def rgb2gray(rgb):
    if len(rgb.shape) == 3:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = rgb
    return gray


def is_png(filename):
    return filename.split('.')[-1] == 'png'


def is_mask(f):
    return f.split("/")[-1].split("_")[0] == 'mask'


def is_gt(f):
    return f.split("/")[-1].split("_")[0] == 'gt'


def is_cage(f):
    return f.split("/")[-1].split("_")[0] == 'cage'


def is_txt(filename):
    return filename.split('.')[-1] == 'txt'


def get_images(files, root):
    images = []
    for f in files:
        if is_png(f) and not is_mask(f) and not is_gt(f):
            image = ImageClass()
            image.read_png(root + "/" + f)
            images.append(image)
    return images


def get_masks(files, root):
    masks = []
    for f in files:
        if is_png(f) and is_mask(f):
            mask = MaskClass()
            mask.read_png(root + "/" + f)
            masks.append(mask)
    return masks


def get_cages(files, root):
    cages = []
    for f in files:
        if is_txt(f) and is_cage(f):
            cage = CageClass()
            cage.read_txt(root + "/" + f)
            cages.append(cage)
    return cages


def get_ground_truth(image, files):
    gt_name = 'gt_' + image.name
    if gt_name in files:
        gt_image = MaskClass()
        gt_image.read_png(image.root + gt_name)
        return gt_image
    else:
        return None


def create_ground_truth(initial_cage, final_cage, initial_mask):
    from ctypes_utils import *

    contour_coord, contour_size = get_contour(initial_mask)
    affine_contour_coordinates = get_affine_contour_coordinates(contour_coord, initial_cage.cage)

    # Update Step of contour coordinates
    contour_coord = np.dot(affine_contour_coordinates, final_cage.cage)
    band_size = int(initial_mask.height)
    omega_1_coord, omega_2_coord, omega_1_size, omega_2_size = get_omega_1_and_2_coord(band_size, contour_coord,
                                                                                       contour_size, initial_mask.width,
                                                                                       initial_mask.height)
    gt_im = np.zeros([initial_mask.height, initial_mask.width])
    # print cc.polygon_comparison(initial_cage.cage, final_cage.cage)

    # Flip the coordinate points
    gt_im[omega_1_coord.transpose().astype(int)[0], omega_1_coord.transpose().astype(int)[1]] = 255.

    gt_mask = MaskClass(gt_im)
    return gt_mask


def sorensen_dice_coefficient(mask1, mask2):
    mask1_bool = mask1.mask == 255.
    mask2_bool = mask2.mask == 255.
    intersection_bool = mask1_bool & mask2_bool
    intersection_card = sum(intersection_bool)
    sorensen_dice = 2 * intersection_card / float(sum(mask1_bool) + sum(mask2_bool))
    return sorensen_dice


def read_png(name):
    '''
    Return image data from a raw PNG file as numpy array.
    :param name: a directory path to the png image.
    :return: An image matrix in the type (y,x)
    '''
    im = Image.open(name)  # scipy.misc.imread(name)
    im = im.convert('RGB').convert('L')
    im = np.array(im)
    im = im.astype(np.float64)
    return im


def printNpArray(im, show_plot=True):
    im_aux = im.astype('uint8')
    plt.gray()
    plt.imshow(im_aux, interpolation='nearest')
    plt.axis('off')
    if show_plot:
        plt.show()


def binarizePgmImage(image):
    im = image.copy()
    if len(im.shape) == 3:
        im = im[:, :, 0]
    for i in xrange(0, im.shape[0]):
        for j in xrange(0, im.shape[1]):
            if im[i, j] >= 125.:
                im[i, j] = 255.
            else:
                im[i, j] = 0.
    return im


def plotContourOnImage(contour_coordinates, image, points=[], color=[255., 255., 255.], points2=[]):
    f = plt.figure()

    matriu = contour_coordinates.astype(int)
    image_copy = np.copy(image)

    if len(image.shape) == 3:
        image_r = image_copy[:, :, 0]
        image_g = image_copy[:, :, 1]
        image_b = image_copy[:, :, 2]
        size = image.shape
    else:
        image_gray = image_copy
        size = image_gray.shape

    for a in matriu:
        if is_inside_image(a, size):
            if len(size) == 3:
                image_r[a[0]][a[1]] = color[0]
                image_g[a[0]][a[1]] = color[1]
                image_b[a[0]][a[1]] = color[2]
            else:
                image_gray[a[0]][a[1]] = 255.

    if len(size) == 3:
        image_copy[:, :, 0] = image_r
        image_copy[:, :, 1] = image_g
        image_copy[:, :, 2] = image_b
    else:
        image_copy = image_gray
    printNpArray(image_copy, False)

    if points != []:
        points = np.fliplr(points)
        points = np.concatenate((points, [points[0]]))
        points = np.transpose(points)
        plt.scatter(points[0], points[1], marker='o', c='b', )
        plt.plot(points[0], points[1])

    if points2 != []:
        points2 = np.fliplr(points2)
        points2 = np.concatenate((points2, [points2[0]]))
        points2 = np.transpose(points2)
        plt.scatter(points2[0], points2[1], marker='o', c='g', )
        plt.plot(points2[0], points2[1])
    plt.show()


def get_inputs(arguments):
    """Return imagage, cage/s and mask from input as numpy array.
        Specification:  ./cac model imatge mascara caixa_init [caixa_curr]
    """
    if len(arguments) != 6 and len(arguments) != 5:
        print 'Wrong Use!!!! Expected Input ' + sys.argv[0] + \
              ' model(int) image(int) mask(int) init_cage(int) [curr_cage(int)]'
        sys.exit(1)

    model = arguments[1]  # Model
    image = arguments[2]  # Image
    mask = arguments[3]
    init_cage_name = arguments[4]

    if len(arguments) == 6:
        curr_cage_name = int(arguments[5])
    else:
        curr_cage_name = None

    mask_name = mask  # Both .pgm as well as png work. png gives you a rbg image!


    # LOAD Cage/s and Mask
    image = read_png(image)
    mask_file = read_png(mask_name)
    mask_file = binarizePgmImage(mask_file)
    init_cage_file = np.loadtxt(init_cage_name, float)
    if curr_cage_name != None:
        curr_cage_file = np.loadtxt(curr_cage_name, float)
    else:
        curr_cage_file = None

    # FROM RGBA to RGB if necessary
    if image.shape == 2 and image.shape[2] == 4:
        image = image[:, :, 0:3]
    return image, mask_file, init_cage_file, curr_cage_file


def evaluate_image(coordinates, image, outside_value=255.):
    '''
    Evaluates image
    :param coordinates: index numpy array
    :param image: numpy array image
    :return:
        Result of image, when indexes are not inside the image return maximum 255
    '''
    image_evaluations = np.ones([1, len(coordinates)]) * outside_value
    image_evaluations = image_evaluations[0]
    coordinates_booleans = are_inside_image(coordinates, image.shape)
    coordinates_aux = coordinates[coordinates_booleans]
    coordinates_aux = np.transpose(coordinates_aux).tolist()
    image_evaluations[coordinates_booleans] = image[coordinates_aux]
    return image_evaluations


def evaluate_bilinear_interpolation(coordinates, image, outside_value=255.):
    '''

    :param coordinates:
    :param image:
    :param outside_value:
    :return:
    '''
    image_with_border = np.insert(image, (0, image.shape[1]), outside_value, axis=1)
    image_with_border = np.insert(image_with_border, (0, image_with_border.shape[0]), 255, axis=0)
    evaluated_values = [bilinear_interpolate(image_with_border, coord[0] + 1, coord[1] + 1) for coord in coordinates]
    return evaluated_values


def cage_out_of_the_picture(coordinates, size):
    '''
    Test if the WHOLE cage, ie. all of the points are out of the image with the specified size.
    :param coordinates:
    :param size:
    :return:
    '''
    are_inside = are_inside_image(coordinates, size)
    return not bool(sum(are_inside))


# Check if list of points are inside an image given only the shape.
def are_inside_image(coordinates, size):
    boolean = (coordinates[:, 0] > -1) & (coordinates[:, 0] < size[0]) & (coordinates[:, 1] > -1) & (
        coordinates[:, 1] < size[1])
    return boolean


# TODO: coordinates[coordinates[:,0]>1] does not work

# Checks if point is inside an image given only the shape of the image.
def is_inside_image(a, size):
    if a[0] >= 0 and a[0] < size[0] and a[1] >= 0 and a[1] < size[1]:
        return True
    else:
        return False


def bilinear_interpolate(im, y, x):
    # TODO:PROPERLY CHANGE COLUMN AND ROWS TO ROW;COL FORMAT INSTEAD OF COL;ROW
    x_aux = np.asarray(x)
    y_aux = np.asarray(y)

    x0 = np.floor(x_aux).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_aux).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y_aux)
    wb = (x1 - x) * (y_aux - y0)
    wc = (x - x0) * (y1 - y_aux)
    wd = (x - x0) * (y_aux - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def walk_level(some_dir, level=1):
    '''
    Walks through directories with a depth of level which is one by default.
    :param some_dir:
    :param level:
    :return root, dirs, files:
        root: filename of a directory
        dirs: list of directories inside the root directory
        files: list of files inside the root directory
    '''
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def save_cage(cage, filename):
    text_file = open(filename, "w")
    for x, y in cage.cage:
        text_file.write("%.8e\t%.8e\n" % (x, y))


def mkdir(str_path):
    """
    Creates the given path if it does not exist.
    If it exits, returns silently.
    If the directory could not be created raises
    an exception.

    :param str_path:
    :return: True if the path was created, false
    if already existed.
    """
    if os.path.exists(str_path) and os.path.isdir(str_path):
        return False

    os.makedirs(str_path)
    if not os.path.exists(str_path) or not os.path.isdir(str_path):
        raise IOError("Could not create path '%s'" % str_path)

    return True