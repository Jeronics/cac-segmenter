# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created: Mon Jan 12 02:07:05 2015
#      by: PyQt5 UI code generator 5.3.2
#
# WARNING! All changes made in this file will be lost!

from main_directory import utils
from PyQt5 import QtCore, QtGui, QtWidgets


class UiForm(object):
    def setupUi(self, formObject):
        formObject.setObjectName(formObject.path+"Click on the center and radius")
        image = utils.read_png(formObject.path)
        formObject.resize(image.shape[1], image.shape[0])
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(formObject.sizePolicy().hasHeightForWidth())
        formObject.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(formObject)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = QtWidgets.QGraphicsView(formObject)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 1, 1)

        self.retranslateUi(formObject)
        QtCore.QMetaObject.connectSlotsByName(formObject)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Click on the center and radius", "Click on the center and radius"))

