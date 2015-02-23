import sys

from PyQt5 import Qt
from PyQt5 import uic
from PyQt5.QtGui import *

a = Qt.QApplication(sys.argv)

from untitled import UiForm

import numpy as np
from main_directory import utils
from PIL import Image

PI = 3.14159265358979323846264338327950288419716939937510


def create_mask_and_cage_points(c, p, in_filename):
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
    num_cage_points = [8, 9]
    radius = np.linalg.norm(np.array(c) - np.array(p))
    radius_cage_ratio = [1.5, 1.7]
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
    utils.plotContourOnImage(np.array(mask_points), im[:, :, 0],
                             points=cages[str(num_cage_points[0]) + '_' + str(radius_cage_ratio[0])],
                             points2=cages[str(num_cage_points[1]) + '_' + str(radius_cage_ratio[1])])


class OverrideGraphicsScene(Qt.QGraphicsScene):
    def __init__(self, parent=None, in_filename=''):
        self.COUNTER = 0
        self.CENTER = []
        self.RADIUS_POINT = []
        self.in_filename = in_filename
        super(OverrideGraphicsScene, self).__init__(parent)

    def mousePressEvent(self, event):
        super(OverrideGraphicsScene, self).mousePressEvent(event)
        position = (event.pos().y(), event.pos().x())

        if self.COUNTER == 0:
            # The first point is the center
            self.CENTER = [event.pos().y(), event.pos().x()]
            self.COUNTER += 1
        else:
            # The second point is a point in the radius
            self.RADIUS_POINT = [event.pos().y(), event.pos().x()]
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


if __name__ == '__main__':

    RootFolder = '../dataset'
    depth = 2
    generator = utils.walk_level(RootFolder, depth)

    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(RootFolder.split("/")) + depth][1:]
    for r, f in gens:
        # function to be called when mouse is clicked
        for files in f:
            if files.split('.')[-1] == 'png' and files.split("/")[-1].split("_")[0] != 'mask':
                in_filename = r + "/" + files
                open_canvas(in_filename)