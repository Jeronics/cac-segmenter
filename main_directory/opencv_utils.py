import numpy as np
import cv2


def erode(mask, width=5):
    kernel = np.ones((width, width), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    return erosion


def dilate(mask,  width=5):
    kernel = np.ones((width, width), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    return dilation
