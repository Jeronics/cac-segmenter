__author__ = 'jeroni'
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import numpy as np
from main_directory import utils

PI = 3.14159265358979323846264338327950288419716939937510


def create_mask_and_cage_points(c, p, image, num_cage_points):
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
    radius_cage_ratio = [1.3, 1.5, 1.7]
    im = np.zeros(im_shape, dtype='uint8')
    print 'Shape', im_shape
    print c
    mask_points = []

    # careful im_shape is (max(y), max(x))
    for x in xrange(im_shape[1]):
        for y in xrange(im_shape[0]):
            if pow(y - c[0], 2) + pow(x - c[1], 2) <= pow(radius, 2):
                im[y, x] = 255
                mask_points.append([y, x])

    ima = Image.fromarray(im)
    folder = '/'.join(IN_FILENAME.split("/")[:-1])
    ima.save(folder + "/mask_00.png")
    print type(num_cage_points) is list
    if type(num_cage_points) is not list:
        num_cage_points = [num_cage_points]
    cages = {}
    for ratio in radius_cage_ratio:
        for n in num_cage_points:
            name_output = folder + '/cage_' + str(n) + '_' + str(ratio) + '.txt'
            text_file = open(name_output, "w")
            print n
            cage = []
            for i in xrange(0, n):
                angle = 2 * i * PI / n
                y, x = radius * ratio * np.sin(angle), radius * ratio * np.cos(angle)
                cage.append([y + c[0], x + c[1]])
                text_file.write("%.8e\t%.8e\n" % (y, x))
            cages[str(n) + '_' + str(ratio)] = np.array(cage)
    print cages.keys()
    utils.plotContourOnImage(np.array(mask_points), image,
                             points=cages[str(num_cage_points[0]) + '_' + str(radius_cage_ratio[0])],
                             points2=cages[str(num_cage_points[1]) + '_' + str(radius_cage_ratio[1])])


class DrawImage(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)

        self.setWindowTitle('Select Window')
        self.local_image = QImage(IN_FILENAME)

        self.local_grview = QGraphicsView()
        self.setCentralWidget(self.local_grview)

        self.local_scene = QGraphicsScene()

        self.image_format = self.local_image.format()
        self.pixMapItem = self.local_scene.addPixmap( QPixmap(self.local_image) )
        # self.pixMapItem = QGraphicsPixmapItem(QPixmap(self.local_image), None, self.local_scene)

        self.local_grview.setScene(self.local_scene)

        self.pixMapItem.mousePressEvent = self.pixelSelect

    def pixelSelect(self, event):
        print 'hello'
        position = QPoint(event.pos().x(), event.pos().y())
        print(str(event.pos().x()))
        print( str(event.pos().y()) )

        global COUNTER
        global CENTER
        global RADIUS_POINT
        global IN_FILENAME
        if COUNTER == 0:
            # The first point is the center
            CENTER = [event.y, event.x]
            print 'Center', CENTER
            COUNTER += 1
        else:
            # The second point is a point in the radius
            RADIUS_POINT = [event.y, event.x]
            print 'RADIUS_POINT', RADIUS_POINT
            create_mask_and_cage_points(CENTER, RADIUS_POINT, image, num_cage_points)


def open_canvas():
    app = QtWidgets.QApplication(sys.argv)
    form = DrawImage()
    form.show()
    app.exec_()


if __name__ == '__main__':

    RootFolder = '../dataset'
    depth = 2
    generator = utils.walk_level(RootFolder, depth)

    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(RootFolder.split("/")) + depth][1:]
    print gens
    for r, f in gens:
        COUNTER = 0
        # function to be called when mouse is clicked
        CENTER = []
        RADIUS_POINT = []
        for files in f:
            if files.split('.')[-1] == 'png':
                print files
                IN_FILENAME = r + "/" + files
                print IN_FILENAME
                open_canvas()

