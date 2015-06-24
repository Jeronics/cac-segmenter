import cv2
import numpy as np


def erode_mask(mask_obj):
    return None


def dilate_mask(mask_obj):
    return None


def numpy_array_to_open_cv(np_array):
    im = np.array(np_array*255, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    return threshed


if __name__ == '__main__':
    np_array=np.random.random((4,4))
    x=numpy_array_to_open_cv(np_array)
    print x