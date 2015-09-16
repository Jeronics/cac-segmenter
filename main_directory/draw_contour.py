__author__ = 'jeronicarandellsaladich'

import numpy as np

import matplotlib.pyplot as plt

from MaskClass import MaskClass
import cv2


def return_contour(mask_obj):
    imgray = mask_obj.mask.astype(np.uint8)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def plot_contour_on_image(image_obj, mask_obj):
    contours = return_contour(mask_obj)
    fig = plt.figure()
    plt.xlim(0, image_obj.shape[0])
    plt.ylim(0, image_obj.shape[1])
    image_obj.plot_image(show_plot=False)
    lon = contours[0][:].T[0][0]
    lat = contours[0][:].T[1][0]
    print contours
    plt.fill(lon, lat, fill=False, color='b', linewidth=2)
    plt.show()

im = np.zeros([300, 200])
for i in xrange(im.shape[0]):
    if i > 50 and i <= 75:
        for j in xrange(im.shape[1]):
            if j > 25 and j <= 65:
                im[i, j] = 255.
mask_obj = MaskClass(mask=im)

plot_contour_on_image(mask_obj, mask_obj)
