import numpy as np
from scipy import *
import os
import sys
from MaskClass import MaskClass
from ImageClass import ImageClass
from CageClass import CageClass
import matplotlib.pyplot as plt


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
    if not omega_1_size:
        print 'Contour has closed in or expanded.'
        return None
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
    intersection_card = np.sum(intersection_bool)
    sorensen_dice = 2 * intersection_card / float(np.sum(mask1_bool) + np.sum(mask2_bool))
    return sorensen_dice


def evaluate_segmentation(mask1, mask2):
    mask1_bool = mask1.mask == 255.
    mask2_bool = mask2.mask == 255.
    c_mask1_bool = mask1.mask == 0.
    c_mask2_bool = mask2.mask == 0.

    # True Positives
    TP_bool = mask1_bool & mask2_bool
    TP = np.sum(TP_bool)

    # True Negatives
    TN_bool = c_mask1_bool & c_mask2_bool
    TN = np.sum(TN_bool)

    # False Positive
    FP_bool = c_mask1_bool & mask2_bool
    FP = np.sum(FP_bool)

    # False Negative
    FN_bool = mask1_bool & c_mask2_bool
    FN = np.sum(FN_bool)
    return TP, TN, FP, FN

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
    coordinates_booleans = are_inside_image(coordinates, image.shape)
    coordinates_aux = coordinates[coordinates_booleans]
    coordinates_aux = np.transpose(coordinates_aux).tolist()
    image_evaluations = image[coordinates_aux]
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
    return_bool= not bool(sum(are_inside))
    return return_bool


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