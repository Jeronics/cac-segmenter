import sys

from PyQt5 import Qt
from PyQt5 import uic
from PyQt5.QtGui import *

a = Qt.QApplication(sys.argv)

# from untitled import Ui_Form

import numpy as np
from main_directory import utils
from PIL import Image

PI = 3.14159265358979323846264338327950288419716939937510

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Click on the center and radius")
        image=utils.read_png(IN_FILENAME)
        print image.shape[0],image.shape[1]
        Form.resize(image.shape[0],image.shape[1])
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Click on the center and radius", "Click on the center and radius"))

# Override like this:
def create_mask_and_cage_points(c, p):
    '''
    This function creates a mask and a sequence of cages.
    :param c:
    :param p:
    :param im_shape:
    :param num_cage_points:
    :return:
    '''
    num_cage_points = [8,9,10]
    image=utils.read_png(IN_FILENAME)
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
    # utils.plotContourOnImage(np.array(mask_points), image,
    #                          points=cages[str(num_cage_points[0]) + '_' + str(radius_cage_ratio[0])],
    #                          points2=cages[str(num_cage_points[1]) + '_' + str(radius_cage_ratio[1])])


class override_graphicsScene(Qt.QGraphicsScene):
    def __init__(self, parent=None):
        super(override_graphicsScene, self).__init__(parent)

    def mousePressEvent(self, event):
        super(override_graphicsScene, self).mousePressEvent(event)
        position = (event.pos().y(), event.pos().x())
        print position

        global COUNTER
        global CENTER
        global RADIUS_POINT
        global IN_FILENAME
        if COUNTER == 0:
            # The first point is the center
            CENTER = [event.pos().y(), event.pos().x()]
            print 'Center', CENTER
            COUNTER += 1
        else:
            # The second point is a point in the radius
            RADIUS_POINT = [event.pos().y(), event.pos().x()]
            print 'RADIUS_POINT', RADIUS_POINT
            create_mask_and_cage_points(CENTER, RADIUS_POINT)


class Image_Process(Qt.QWidget):
    def __init__(self):
        super(Image_Process, self).__init__()
        self.path = IN_FILENAME  # image path
        print QImage(self.path).size()
        print QImage(self.path).width(), QImage(self.path).height()
        self.new = Ui_Form()
        self.new.setupUi(self)

        self.pixmap = Qt.QPixmap()
        self.pixmap.load(self.path)
        self.pixmap = self.pixmap.scaled(self.size(), Qt.Qt.KeepAspectRatio)

        self.graphicsPixmapItem = Qt.QGraphicsPixmapItem(self.pixmap)

        self.graphicsScene = override_graphicsScene(self)
        self.graphicsScene.addItem(self.graphicsPixmapItem)

        self.new.graphicsView.setScene(self.graphicsScene)


def open_canvas():
    my_Qt_Program = Image_Process()
    my_Qt_Program.show()
    a.exec_()


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
            if files.split('.')[-1] == 'png' and files.split("/")[-1].split("_")[0]!='mask':
                print files
                IN_FILENAME = r + "/" + files
                print IN_FILENAME
                open_canvas()