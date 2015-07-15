import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

image_path = '../../BSDS300/images/other/colorful_balloons.png'
balloons = utils.ImageClass()
balloons.read_png(image_path)
hsi = balloons.hsi_image[:, :, 0] / (2 * 3.14) * 360.
# azi = azi.flatten()

plt.imshow(hsi, cmap=matplotlib.cm.hsv)
plt.axis('off')
plt.show()

im = np.zeros([300, 400, 3])

balloons = utils.ImageClass(im=im)
hsi = balloons.hsi_image[:, :, 0] / (2 * 3.14) * 360.
# azi = azi.flatten()
balloons.plot_hsi_image()
plt.imshow(hsi, cmap=matplotlib.cm.hsv)
plt.axis('off')
plt.show()