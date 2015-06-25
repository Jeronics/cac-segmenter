import cv2
import numpy as np
import cv

def erode_mask(mask_obj):
    return None


def dilate_mask(mask_obj):
    return None


def numpy_array_to_open_cv(np_array):
    im = np.array(np_array * 255, dtype=np.uint8)
    threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    return threshed

def np_to_array(vis):
    h,w = vis.shape
    vis2 = cv.CreateMat(h, w, cv.CV_32FC3)
    vis0 = cv.fromarray(vis)
    cv.CvtColor(vis0, vis2, cv.CV_GRAY2BGR)
    return vis0




if __name__ == '__main__':
    ImageClass
    np_array = np.ones((384, 836), np.float32)
    x=np_to_array(np_array)
    import pdb; pdb.set_trace()