__author__ = 'jeroni'
from main_directory import utils as ut
import numpy as np
import cv2

im_obj = ut.MaskClass()
im_obj.read_png('eagle/eagle1/mask_00.png')
im_obj.plot_image()

image = cv2.imread('eagle/eagle1/mask_00.png', cv2.CV_8UC1)  # Load your image in here
# Your code to threshold
# image_obj = ut.MaskClass()
# image.read_png('')
print image

image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 10)

# Perform morphology
se = np.ones((7, 7), dtype='uint8')
image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

# Your code now applied to the closed image
cnt = cv2.findContours(image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
mask = np.zeros(image.shape[:2], np.uint8)
print mask
# cv2.drawContours(mask, cnt, -1, 255, -1)