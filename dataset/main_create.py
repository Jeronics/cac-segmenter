import sys

from PyQt5 import Qt
from PyQt5 import uic
from PyQt5.QtGui import *

a = Qt.QApplication(sys.argv)

from untitled import UiForm

import numpy as np
from main_directory import utils
from PIL import Image

from polygon_tools import turning_function

PI = 3.14159265358979323846264338327950288419716939937510


def create_mask_and_cage_points(c, p, in_filename, display_mask=False):
    '''
    This function creates a mask and a sequence of cages.
    :param c:
    :param p:
    :param im_shape:
    :param num_cage_points:
    :return:
    '''
    image = utils.read_png(in_filename)
    im_shape = image.shape
    num_cage_points = [16]
    radius = np.linalg.norm(np.array(c) - np.array(p))
    radius_cage_ratio = [1.05]
    im = np.zeros(im_shape, dtype='uint8')
    mask_points = []

    # careful im_shape is (max(y), max(x))
    for x in xrange(im_shape[1]):
        for y in xrange(im_shape[0]):
            if pow(y - c[0], 2) + pow(x - c[1], 2) <= pow(radius, 2):
                im[y, x] = 255
                mask_points.append([y, x])

    ima = Image.fromarray(im[:, :, 0], mode='L')
    folder = '/'.join(in_filename.split("/")[:-1])
    ima.save(folder + "/mask_00.png")
    if type(num_cage_points) is not list:
        num_cage_points = [num_cage_points]
    cages = {}
    for ratio in radius_cage_ratio:
        for n in num_cage_points:
            name_output = folder + '/cage_' + str(n) + '_' + str(ratio) + '.txt'
            text_file = open(name_output, "w")
            cage = []
            for i in xrange(0, n):
                angle = 2 * i * PI / n
                y, x = radius * ratio * np.sin(angle), radius * ratio * np.cos(angle)
                cage.append([y + c[0], x + c[1]])
                text_file.write("%.8e\t%.8e\n" % (x + c[1], y + c[0]))  # OTHER
            cages[str(n) + '_' + str(ratio)] = np.array(cage)
    if display_mask:
        utils.plotContourOnImage(np.array(mask_points), im[:, :, 0],
                                 points=cages[str(num_cage_points[0]) + '_' + str(radius_cage_ratio[0])])
    a.closeAllWindows()


class OverrideGraphicsScene(Qt.QGraphicsScene):
    def __init__(self, parent=None, in_filename=''):
        self.COUNTER = 0
        self.CENTER = []
        self.RADIUS_POINT = []
        self.in_filename = in_filename
        super(OverrideGraphicsScene, self).__init__(parent)

    def mousePressEvent(self, event):
        super(OverrideGraphicsScene, self).mousePressEvent(event)
        position = (event.pos().x(), event.pos().y())
        print '(y,x)=', event.pos().x(), ',', event.pos().y()
        if self.COUNTER == 0:
            # The first point is the center
            self.CENTER = [event.pos().x(), event.pos().y()]
            self.COUNTER += 1
        else:
            # The second point is a point in the radius
            self.RADIUS_POINT = [event.pos().x(), event.pos().y()]
            create_mask_and_cage_points(self.CENTER, self.RADIUS_POINT, self.in_filename)


class ImageProcess(Qt.QWidget):
    def __init__(self, in_filename):
        super(ImageProcess, self).__init__()
        self.path = in_filename  # image path
        self.new = UiForm()
        self.new.setupUi(self)

        self.pixmap = Qt.QPixmap()
        self.pixmap.load(self.path)
        self.pixmap = self.pixmap.scaled(self.size(), Qt.Qt.KeepAspectRatio)

        self.graphicsPixmapItem = Qt.QGraphicsPixmapItem(self.pixmap)

        self.graphicsScene = OverrideGraphicsScene(self, in_filename=in_filename)
        self.graphicsScene.addItem(self.graphicsPixmapItem)

        self.new.graphicsView.setScene(self.graphicsScene)


def open_canvas(in_filename):
    my_Qt_Program = ImageProcess(in_filename)
    my_Qt_Program.show()
    a.exec_()


def resize_image(image):
    if image.height > 500:
        image.reshape(new_width=400)
        image.save_image(filename=image.path)


def create_ground_truth(image):
    if isinstance(image, utils.ImageClass):
        im = image.gray_image
        ground_truth = utils.MaskClass(mask=im, filename=image.root + 'gt_' + image.spec_name + '.png', threshold=252.)
        ground_truth.save_image(filename=ground_truth.path)

def resize_mask(mask):
    print mask.name
    if mask.height > 500:
        print 'DO'
        mask.reshape(new_width=400)
        mask.save_image(filename=mask.path)




if __name__ == '__main__':
    RootFolder = '../dataset'
    depth = 2
    generator = utils.walk_level(RootFolder, depth)

    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(RootFolder.split("/")) + depth]
    for root, files in gens:
        #
        # # All images in each file are found
        # cages = utils.get_cages(files, root)
        # for cage in cages:
        #     turning_function.plot_polygon(cage.cage, fig_title=cage.root)

        # All images in each file are found
        images = utils.get_images(files, root)
        for image in images:
            # if image.spec_name == 'star5':
            print image.spec_name
            open_canvas(image.path)
            # resize_image(image)
            # gt=create_ground_truth(image)
            # open_canvas(image.path)

        # # All masks in each file are found
        # images = utils.get_images(files,root)
        # for image in images:
        #     gt_im=utils.get_ground_truth(image,files)
        #     if gt_im:
        #         resize_mask(gt_im)
