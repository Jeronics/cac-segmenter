from ImageClass import ImageClass
from MaskClass import MaskClass
import numpy as np

if __name__ == '__main__':
    folder_name = '../../1obj/100_0109/human_seg/'
    mask_file = '100_0109_7.png'
    filename = folder_name + mask_file
    image = ImageClass()
    image.read_png(filename)
    image.plot_image()
    mask = image.image
    a = [255., 0., 0.]
    mask[mask == a] = 255.
    mask[mask != a] = 0.
    m = MaskClass()
    m.mask = mask[:, :, 0]
    m.plot_image()
